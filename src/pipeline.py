import base64
import queue
import socket
import subprocess
import threading
import time
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import zoom

from .audio import YAMNetAnalyser, audio_db, build_audio_ffmpeg_cmd, combine_life_signs
from .config import (
    AUDIO_ANALYSIS_INTERVAL_S,
    AUDIO_ANALYSIS_WINDOW_S,
    AUDIO_CHUNK_S,
    AUDIO_MAX_STALE_MS,
    AUDIO_SAMPLE_RATE,
    AUDIO_STATUS_INTERVAL_S,
    AUDIO_STREAM_URL,
    BUFFER_SIZE,
    FRAME_QUEUE_SIZE,
    FUSION_MAX_DT_MS,
    IR_BOOST,
    IR_HUMAN_MAX_C,
    IR_HUMAN_MIN_C,
    PHONE_STREAM_URL,
    SOCKET_RCVBUF,
    UDP_HOST,
    UDP_PORT_CAM,
    UDP_PORT_IR,
    log,
)
from .models import AudioState, CamFrame, IRFrame
from .packets import JpegReassembler, parse_cam_hdr, parse_ir
from .vision import IRAnalyser, PersonTracker, YOLODetector, suppress_duplicate_persons
from .websocket_server import WSBroadcaster


class Pipeline:
    def __init__(self, broadcaster: WSBroadcaster):
        self.broadcaster = broadcaster
        self.yolo = YOLODetector()
        self.tracker = PersonTracker()
        self.ir_anal = IRAnalyser()
        self.audio_anal = YAMNetAnalyser()
        self._shared: dict = {}  # {"ir": IRFrame, "audio": AudioState}
        self._frame_q: queue.Queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self._sent_frames = 0

    def start(self):
        # Always launch IR loop
        threading.Thread(target=self._ir_loop, daemon=True, name="IR-Recv").start()

        # Branch video receiving strategies based on config
        if PHONE_STREAM_URL is not None:
            threading.Thread(
                target=self._phone_stream_loop, daemon=True, name="STREAM-Recv"
            ).start()
            threading.Thread(
                target=self._phone_audio_loop, daemon=True, name="STREAM-Audio"
            ).start()
        else:
            threading.Thread(
                target=self._cam_loop, daemon=True, name="CAM-Recv"
            ).start()

        threading.Thread(
            target=self._yolo_loop, daemon=True, name="YOLO-Worker"
        ).start()
        log.info(f"[Pipeline] Threads started. IR Listening on UDP:{UDP_PORT_IR}")

    # ── IR receiver (Unchanged - collects from port 5005) ────────────────────
    def _ir_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_RCVBUF)
        sock.bind((UDP_HOST, UDP_PORT_IR))
        sock.settimeout(2.0)
        log.info(
            f"[IR ] Listening on {UDP_HOST}:{UDP_PORT_IR}  rcvbuf={SOCKET_RCVBUF//1024}KB"
        )

        while True:
            try:
                data, _ = sock.recvfrom(BUFFER_SIZE)
                frame = parse_ir(data)
                if frame is None:
                    continue

                self._shared["ir"] = frame

                self.broadcaster.broadcast_json(
                    {
                        "type": "ir",
                        "ir_grid": frame.temps.flatten().tolist(),
                    }
                )
            except socket.timeout:
                pass
            except Exception as e:
                log.error(f"[IR ] {e}")

    # ── Phone Stream Receiver (NEW HTTP/RTSP Thread) ─────────────────────────
    def _phone_stream_loop(self):
        log.info(f"[STREAM] Connecting to phone camera stream: {PHONE_STREAM_URL}")

        stream_frames = 0

        while True:
            cap = cv2.VideoCapture(PHONE_STREAM_URL)
            if not cap.isOpened():
                log.error("[STREAM] Failed to open stream. Retrying in 3 seconds...")
                time.sleep(3)
                continue

            log.info("[STREAM] Connected successfully to phone camera feed.")

            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    log.warning(
                        "[STREAM] Lost connection to stream. Attempting reconnect..."
                    )
                    break

                stream_frames += 1

                # Mock a CamFrame object out of the raw frame data
                ts = time.monotonic_ns() // 1000
                seq = stream_frames & 0xFFFFFFFF

                # Hand it over as a raw image to skip imdecode inside the YOLO worker
                cam = CamFrame(seq=seq, timestamp=ts, jpeg=b"", raw_img=frame_bgr)
                ir_now = self._shared.get("ir")

                try:
                    self._frame_q.put_nowait((cam, ir_now))
                except queue.Full:
                    try:
                        self._frame_q.get_nowait()
                    except queue.Empty:
                        pass
                    self._frame_q.put_nowait((cam, ir_now))

            cap.release()
            time.sleep(1)

    # ── Phone audio receiver (FFmpeg PCM + optional YAMNet) ──────────────────
    def _phone_audio_loop(self):
        if not AUDIO_STREAM_URL:
            return

        chunk_samples = max(1, int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_S))
        chunk_bytes = chunk_samples * 2
        analysis_samples = max(
            chunk_samples, int(AUDIO_SAMPLE_RATE * AUDIO_ANALYSIS_WINDOW_S)
        )
        audio_buffer = np.empty(0, dtype=np.float32)
        last_analysis = 0.0
        last_status = 0.0

        audio_url = AUDIO_STREAM_URL
        cmd = build_audio_ffmpeg_cmd(audio_url)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            log.warning("[AUDIO] FFmpeg not found - phone stream audio disabled")
            return
        except Exception as e:
            log.warning(f"[AUDIO] Failed to start FFmpeg audio reader: {e}")
            return

        log.info(f"[AUDIO] Connected to phone camera audio feed: {audio_url}")

        try:
            while proc.stdout:
                data = proc.stdout.read(chunk_bytes)
                if not data:
                    log.warning(
                        "[AUDIO] Audio stream ended; not opening another connection."
                    )
                    return

                pcm_i16 = np.frombuffer(data, dtype=np.int16)
                if pcm_i16.size == 0:
                    continue

                waveform = pcm_i16.astype(np.float32) / 32768.0
                audio_buffer = np.concatenate((audio_buffer, waveform))
                if audio_buffer.size > analysis_samples * 2:
                    audio_buffer = audio_buffer[-analysis_samples * 2 :]

                state = AudioState(db=audio_db(waveform), confidence=0.0, labels=[])
                now = time.monotonic()
                if (
                    audio_buffer.size >= analysis_samples
                    and now - last_analysis >= AUDIO_ANALYSIS_INTERVAL_S
                ):
                    state = self.audio_anal.analyse(audio_buffer[-analysis_samples:])
                    self._shared["audio"] = state
                    last_analysis = now
                else:
                    prev = self._shared.get("audio")
                    if (
                        isinstance(prev, AudioState)
                        and now - prev.received < AUDIO_MAX_STALE_MS / 1000.0
                    ):
                        state = AudioState(
                            db=state.db, confidence=prev.confidence, labels=prev.labels
                        )
                        self._shared["audio"] = state

                self.broadcaster.broadcast_json(
                    {
                        "type": "audio",
                        "codec": "pcm_s16le",
                        "source_codec": "wav",
                        "source_url": audio_url,
                        "sample_rate": AUDIO_SAMPLE_RATE,
                        "pcm": base64.b64encode(pcm_i16.tobytes()).decode("ascii"),
                        "audio_db": round(state.db, 1),
                        "human_sound": state.human_related,
                        "human_audio_confidence": round(state.confidence, 4),
                        "human_audio_labels": state.labels,
                    }
                )

                if now - last_status >= AUDIO_STATUS_INTERVAL_S:
                    log.info(
                        f"[AUDIO] Receiving audio from {audio_url} ({len(data)} bytes, {state.db:.1f} dB)"
                    )
                    last_status = now
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=1)
            except Exception:
                pass

    # ── Legacy Camera receiver (Used only if PHONE_STREAM_URL is None) ───────
    def _cam_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_RCVBUF)
        reassembler = JpegReassembler()
        sock.bind((UDP_HOST, UDP_PORT_CAM))
        sock.settimeout(2.0)
        log.info(
            f"[CAM] Listening on {UDP_HOST}:{UDP_PORT_CAM}  rcvbuf={SOCKET_RCVBUF//1024}KB"
        )

        dropped_os = 0
        cam_packets = 0
        cam_frames = 0
        bad_packets = 0

        while True:
            try:
                data, _ = sock.recvfrom(BUFFER_SIZE)
                cam_packets += 1
                parsed = parse_cam_hdr(data)
                if parsed is None:
                    bad_packets += 1
                    continue

                seq, ts, tot_len, chunk = parsed
                cam = reassembler.feed(seq, ts, tot_len, chunk)
                if cam is None:
                    continue
                cam_frames += 1

                ir_now = self._shared.get("ir")

                try:
                    self._frame_q.put_nowait((cam, ir_now))
                except queue.Full:
                    try:
                        self._frame_q.get_nowait()
                    except queue.Empty:
                        pass
                    self._frame_q.put_nowait((cam, ir_now))
                    dropped_os += 1
            except socket.timeout:
                pass
            except Exception as e:
                log.error(f"[CAM] {e}")

    # ── YOLO worker (runs YOLO + IR fusion + broadcasts) ─────────────────────
    def _yolo_loop(self):
        log.info("[YOLO] Worker thread ready")
        while True:
            try:
                cam, ir = self._frame_q.get(timeout=2.0)
            except queue.Empty:
                continue

            if ir and (time.monotonic() - ir.received) * 1000 > FUSION_MAX_DT_MS:
                ir = None

            try:
                self._process_frame(cam, ir)
            except Exception as e:
                log.exception(f"[YOLO] frame processing failed: {e}")

    # ── Frame processing (YOLO + annotate + broadcast) ───────────────────────
    def _process_frame(self, cam: CamFrame, ir: Optional[IRFrame]):
        # Handle phone frames (raw numpy matrix) vs ESP32 frames (jpeg encoded bytearray)
        if cam.raw_img is not None:
            img = cam.raw_img.copy()
        else:
            arr = np.frombuffer(cam.jpeg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            log.warning(f"[YOLO] Frame context resolution failed for seq={cam.seq}")
            return

        h, w = img.shape[:2]
        raw_dets = suppress_duplicate_persons(self.yolo.detect_persons(img))
        tracked_dets = self.tracker.update(raw_dets)
        detections = []
        audio = self._shared.get("audio")
        if (
            isinstance(audio, AudioState)
            and (time.monotonic() - audio.received) * 1000 > AUDIO_MAX_STALE_MS
        ):
            audio = None

        for bbox, yolo_conf, track_id in tracked_dets:
            x1, y1, x2, y2 = bbox
            ir_confirmed = False
            ir_max_temp = 0.0
            ir_hot_px = 0
            fused_conf = yolo_conf

            if ir is not None:
                ir_confirmed, ir_max_temp, ir_hot_px = self.ir_anal.analyse(
                    bbox, (h, w), ir
                )
                fused_conf = min(
                    1.0, yolo_conf * (IR_BOOST if ir_confirmed else IR_HUMAN_MAX_C)
                )

            alive_conf, vital_status = combine_life_signs(
                yolo_conf, ir_confirmed, audio
            )

            bx, by, bw, bh = x1, y1, x2 - x1, y2 - y1
            detections.append(
                {
                    "label": "person",
                    "confidence": round(fused_conf, 4),
                    "track_id": track_id,
                    "bbox": [bx, by, bw, bh],
                    "ir_confirmed": ir_confirmed,
                    "ir_max_temp": round(ir_max_temp, 1),
                    "ir_hot_pixels": ir_hot_px,
                    "alive_confidence": round(alive_conf, 4),
                    "vital_status": vital_status,
                    "human_sound": bool(audio and audio.human_related),
                    "human_audio_confidence": (
                        round(audio.confidence, 4) if audio else 0.0
                    ),
                    "human_audio_labels": audio.labels if audio else [],
                }
            )

        annotated = self._annotate(img, detections, ir)
        ok, jpeg_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")

        self.broadcaster.broadcast_json(
            {
                "type": "frame",
                "data": b64,
                "detections": detections,
                "audio": {
                    "db": round(audio.db, 1) if audio else None,
                    "human_sound": bool(audio and audio.human_related),
                    "human_audio_confidence": (
                        round(audio.confidence, 4) if audio else 0.0
                    ),
                    "human_audio_labels": audio.labels if audio else [],
                },
            }
        )
        self._sent_frames += 1

    def _annotate(
        self, img: np.ndarray, detections: list, ir: Optional[IRFrame]
    ) -> np.ndarray:
        vis = img.copy()

        if ir is not None:
            t_min, t_max = ir.temps.min(), ir.temps.max()
            norm = ((ir.temps - t_min) / max(t_max - t_min, 1e-3) * 255).astype(
                np.uint8
            )
            up = zoom(norm, 10, order=1).astype(np.uint8)
            heatmap = cv2.applyColorMap(up, cv2.COLORMAP_INFERNO)
            oh, ow = heatmap.shape[:2]
            h, w = vis.shape[:2]
            xo, yo = w - ow - 4, 4
            roi = vis[yo : yo + oh, xo : xo + ow]
            vis[yo : yo + oh, xo : xo + ow] = cv2.addWeighted(roi, 0.3, heatmap, 0.7, 0)
            cv2.rectangle(vis, (xo, yo), (xo + ow, yo + oh), (80, 80, 80), 1)
            cv2.putText(
                vis,
                "IR",
                (xo + 2, yo + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (220, 220, 220),
                1,
            )
            for r in range(8):
                for c in range(8):
                    if IR_HUMAN_MIN_C <= ir.temps[r, c] <= IR_HUMAN_MAX_C:
                        cv2.drawMarker(
                            vis,
                            (xo + c * 10 + 5, yo + r * 10 + 5),
                            (0, 255, 120),
                            cv2.MARKER_CROSS,
                            5,
                            1,
                        )

        for d in detections:
            bx, by, bw, bh = d["bbox"]
            col = (0, 230, 60) if d["ir_confirmed"] else (0, 180, 255)
            vital = (
                f" LIFE {d['alive_confidence']:.0%}"
                if d["vital_status"] != "inconclusive"
                else " LIFE ?"
            )
            label = f"HUMAN #{d['track_id']} {d['confidence']:.0%}{vital}" + (
                " +IR" if d["ir_confirmed"] else ""
            )
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), col, 2)
            cv2.putText(
                vis,
                label,
                (bx + 2, by - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                col,
                1,
                cv2.LINE_AA,
            )
            if d["ir_confirmed"]:
                cv2.putText(
                    vis,
                    f"{d['ir_max_temp']:.1f}°C ({d['ir_hot_pixels']}px)",
                    (bx + 2, by + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (100, 255, 180),
                    1,
                    cv2.LINE_AA,
                )

        return vis


# ─────────────────────────────────────────────
