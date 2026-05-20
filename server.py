"""
ws_bridge.py  —  ROVER Dashboard WebSocket Bridge
===================================================
Sits between the Python UDP backend and the ROVER frontend (index.html).

  ESP32-CAM/AMG8833
        │  UDP
        ▼
  python_backend.py  (ModelDispatcher)
        │  internal callback
        ▼
  ws_bridge.py  ←── this file
        │  WebSocket (ws://host:8765/ws)
        ▼
  ROVER index.html  (app.js)

Message types sent to the frontend (matches app.js protocol exactly):

  { "type": "frame",
    "data": "<base64 JPEG>",
    "detections": [
      { "label": "person",
        "confidence": 0.87,
        "bbox": [x, y, w, h],
        "ir_confirmed": true,
        "ir_max_temp": 35.2,
        "ir_hot_pixels": 4 }
    ]
  }

  { "type": "ir",
    "ir_grid": [64 floats]          # row-major, °C
  }

  { "type": "audio_level",
    "audio_db": -32.5               # placeholder; wire real audio if needed
  }

Install:
    pip install websockets numpy opencv-python scipy ultralytics

Usage:
    # In one terminal:
    python ws_bridge.py

    # The ROVER frontend Settings:
    #   WebSocket Host: <this machine IP>
    #   Port: 8765
    #   Path: /ws
"""

import asyncio
import base64
import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from scipy.ndimage import zoom

# ── WebSocket (stdlib-free fallback path if websockets not installed) ──────────
try:
    import websockets
    from websockets.server import serve as ws_serve
    _HAS_WEBSOCKETS = True
except ImportError:
    _HAS_WEBSOCKETS = False
    print("[WARN] 'websockets' not installed — run: pip install websockets")

# ─────────────────────────────────────────────
#  CONFIG — edit these to match your setup
# ─────────────────────────────────────────────
UDP_HOST       = "0.0.0.0"
UDP_PORT_IR    = 5005
UDP_PORT_CAM   = 5006
WS_HOST        = "0.0.0.0"
WS_PORT        = 8765          # frontend Settings → Port
WS_PATH        = "/ws"         # frontend Settings → Endpoint Path
MAGIC          = 0x45533332    # "ES32"
BUFFER_SIZE    = 65535

# YOLO
YOLO_WEIGHTS     = "yolov8n.pt"
YOLO_CONF_THRESH = 0.40
COCO_PERSON_ID   = 0

# IR thresholds
IR_HUMAN_MIN_C    = 28.0
IR_HUMAN_MAX_C    = 40.0
IR_MIN_HOT_PIXELS = 2
IR_BOOST          = 1.30
IR_PENALTY        = 0.60

# Fusion
FUSION_MAX_DT_MS = 300

# Broadcast rate-limit: don't push more than N frames/s to WS clients
MAX_BROADCAST_FPS = 12

LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ws_bridge")


# ─────────────────────────────────────────────
#  DATA CLASSES  (same as python_backend.py)
# ─────────────────────────────────────────────
@dataclass
class IRFrame:
    seq:       int
    timestamp: int
    temps:     np.ndarray    # (8, 8) float32, °C
    received:  float = field(default_factory=time.time)


@dataclass
class CamFrame:
    seq:       int
    timestamp: int
    jpeg:      bytes
    received:  float = field(default_factory=time.time)


# ─────────────────────────────────────────────
#  JPEG REASSEMBLER
# ─────────────────────────────────────────────
class JpegReassembler:
    def __init__(self):
        self._buf: Dict[int, bytearray] = {}
        self._tgt: Dict[int, int]       = {}
        self._off: Dict[int, int]       = {}

    def feed(self, seq, timestamp, total_len, chunk) -> Optional[CamFrame]:
        if seq not in self._buf:
            self._buf[seq] = bytearray(total_len)
            self._tgt[seq] = total_len
            self._off[seq] = 0
        offset = self._off[seq]
        end    = offset + len(chunk)
        if end > total_len:
            self._clean(seq); return None
        self._buf[seq][offset:end] = chunk
        self._off[seq] = end
        if end >= total_len:
            frame = CamFrame(seq=seq, timestamp=timestamp, jpeg=bytes(self._buf[seq]))
            self._clean(seq)
            return frame
        return None

    def _clean(self, seq):
        self._buf.pop(seq, None); self._tgt.pop(seq, None); self._off.pop(seq, None)


# ─────────────────────────────────────────────
#  PACKET PARSERS
# ─────────────────────────────────────────────
def parse_ir(data: bytes) -> Optional[IRFrame]:
    if len(data) < 145: return None
    if struct.unpack_from(">I", data, 0)[0] != MAGIC or data[4] != 0x01: return None
    seq = struct.unpack_from(">I", data, 5)[0]
    ts  = struct.unpack_from(">Q", data, 9)[0]
    raw = struct.unpack_from(">64h", data, 17)
    return IRFrame(seq=seq, timestamp=ts, temps=np.array(raw, dtype=np.float32).reshape(8,8) / 100.0)


def parse_cam_hdr(data: bytes):
    if len(data) < 21: return None
    if struct.unpack_from(">I", data, 0)[0] != MAGIC or data[4] != 0x02: return None
    seq     = struct.unpack_from(">I", data, 5)[0]
    ts      = struct.unpack_from(">Q", data, 9)[0]
    tot_len = struct.unpack_from(">I", data, 17)[0]
    return seq, ts, tot_len, data[21:]


# ─────────────────────────────────────────────
#  YOLO DETECTOR
# ─────────────────────────────────────────────
class YOLODetector:
    def __init__(self):
        try:
            from ultralytics import YOLO
            log.info(f"[YOLO] Loading {YOLO_WEIGHTS}")
            self._model   = YOLO(YOLO_WEIGHTS)
            self._backend = "ultralytics"
        except ImportError:
            log.warning("[YOLO] ultralytics not found, using OpenCV DNN fallback")
            self._model   = None
            self._backend = "none"

    def detect_persons(self, img: np.ndarray) -> List[Tuple[Tuple[int,int,int,int], float]]:
        if self._backend == "ultralytics":
            results = self._model(img, conf=YOLO_CONF_THRESH,
                                  classes=[COCO_PERSON_ID], verbose=False)
            out = []
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    out.append(((x1,y1,x2,y2), conf))
            return out
        return []


# ─────────────────────────────────────────────
#  IR ANALYSER
# ─────────────────────────────────────────────
class IRAnalyser:
    def analyse(self, bbox, img_shape, ir: IRFrame):
        img_h, img_w = img_shape
        x1,y1,x2,y2 = [max(0, v) for v in bbox]
        x2 = min(img_w, x2); y2 = min(img_h, y2)

        c1 = max(0, int(x1/img_w*8)); c2 = min(7, int(x2/img_w*8))
        r1 = max(0, int(y1/img_h*8)); r2 = min(7, int(y2/img_h*8))
        if c1 == c2: c2 = min(7, c2+1)
        if r1 == r2: r2 = min(7, r2+1)

        region = ir.temps[r1:r2+1, c1:c2+1]
        if region.size == 0: return False, 0.0, 0

        max_t  = float(region.max())
        hot_px = int(np.sum((region >= IR_HUMAN_MIN_C) & (region <= IR_HUMAN_MAX_C)))
        return hot_px >= IR_MIN_HOT_PIXELS, max_t, hot_px


# ─────────────────────────────────────────────
#  WEBSOCKET BROADCASTER
#  Holds the set of connected WS clients and provides
#  a thread-safe broadcast_json() method callable from
#  non-async threads.
# ─────────────────────────────────────────────
class WSBroadcaster:
    def __init__(self):
        self._clients: Set = set()
        self._lock         = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_broadcast = 0.0
        self._min_interval   = 1.0 / MAX_BROADCAST_FPS

    def set_loop(self, loop):
        self._loop = loop

    def register(self, ws):
        with self._lock:
            self._clients.add(ws)
        log.info(f"[WS] Client connected ({len(self._clients)} total)")

    def unregister(self, ws):
        with self._lock:
            self._clients.discard(ws)
        log.info(f"[WS] Client disconnected ({len(self._clients)} remaining)")

    def broadcast_json(self, payload: dict):
        """Thread-safe: schedule a broadcast onto the asyncio event loop."""
        if not self._loop:
            return
        # Rate-limit
        now = time.monotonic()
        if now - self._last_broadcast < self._min_interval:
            return
        self._last_broadcast = now

        with self._lock:
            clients = set(self._clients)
        if not clients:
            return

        msg = json.dumps(payload)
        asyncio.run_coroutine_threadsafe(self._do_broadcast(msg, clients), self._loop)

    async def _do_broadcast(self, msg: str, clients: set):
        if not clients:
            return
        import websockets.exceptions
        dead = set()
        for ws in clients:
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        if dead:
            with self._lock:
                self._clients -= dead


# ─────────────────────────────────────────────
#  PIPELINE  (UDP receivers + YOLO + IR fusion)
# ─────────────────────────────────────────────
class Pipeline:
    def __init__(self, broadcaster: WSBroadcaster):
        self.broadcaster = broadcaster
        self.yolo        = YOLODetector()
        self.ir_anal     = IRAnalyser()
        self._shared: dict = {}         # {"ir": IRFrame}
        self._ir_thread  = None
        self._cam_thread = None

    def start(self):
        self._ir_thread  = threading.Thread(target=self._ir_loop,  daemon=True, name="IR-Recv")
        self._cam_thread = threading.Thread(target=self._cam_loop, daemon=True, name="CAM-Recv")
        self._ir_thread.start()
        self._cam_thread.start()
        log.info(f"[Pipeline] UDP receivers started (IR:{UDP_PORT_IR} CAM:{UDP_PORT_CAM})")

    # ── IR receiver ───────────────────────────────────────────────────────────
    def _ir_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_HOST, UDP_PORT_IR))
        sock.settimeout(2.0)
        log.info(f"[IR ] Listening on {UDP_HOST}:{UDP_PORT_IR}")

        while True:
            try:
                data, _ = sock.recvfrom(BUFFER_SIZE)
                frame   = parse_ir(data)
                if frame is None:
                    continue

                self._shared["ir"] = frame

                # Forward 8×8 grid to frontend
                self.broadcaster.broadcast_json({
                    "type":    "ir",
                    "ir_grid": frame.temps.flatten().tolist(),
                })

                # Log hotspot
                hot  = float(frame.temps.max())
                warm = int(np.sum((frame.temps >= IR_HUMAN_MIN_C) & (frame.temps <= IR_HUMAN_MAX_C)))
                log.debug(f"[IR ] seq={frame.seq} max={hot:.1f}C warm_px={warm}")

            except socket.timeout:
                pass
            except Exception as e:
                log.error(f"[IR ] {e}")

    # ── Camera receiver ───────────────────────────────────────────────────────
    def _cam_loop(self):
        sock        = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        reassembler = JpegReassembler()
        sock.bind((UDP_HOST, UDP_PORT_CAM))
        sock.settimeout(2.0)
        log.info(f"[CAM] Listening on {UDP_HOST}:{UDP_PORT_CAM}")

        while True:
            try:
                data, _ = sock.recvfrom(BUFFER_SIZE)
                parsed  = parse_cam_hdr(data)
                if parsed is None:
                    continue

                seq, ts, tot_len, chunk = parsed
                cam = reassembler.feed(seq, ts, tot_len, chunk)
                if cam is None:
                    continue    # still collecting fragments

                # Get latest IR for fusion
                ir: Optional[IRFrame] = self._shared.get("ir")
                if ir and abs(cam.timestamp - ir.timestamp) / 1000 > FUSION_MAX_DT_MS:
                    ir = None

                self._process_frame(cam, ir)

            except socket.timeout:
                pass
            except Exception as e:
                log.error(f"[CAM] {e}")

    def _process_frame(self, cam: CamFrame, ir: Optional[IRFrame]):
        # Decode JPEG
        arr = np.frombuffer(cam.jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return

        h, w = img.shape[:2]

        # ── YOLO detection ────────────────────────────────────────────────────
        raw_dets = self.yolo.detect_persons(img)

        detections_out = []
        for (bbox, yolo_conf) in raw_dets:
            x1,y1,x2,y2 = bbox
            ir_confirmed = False
            ir_max_temp  = 0.0
            ir_hot_px    = 0
            fused_conf   = yolo_conf

            if ir is not None:
                ir_confirmed, ir_max_temp, ir_hot_px = self.ir_anal.analyse(bbox, (h,w), ir)
                fused_conf = min(1.0, yolo_conf * (IR_BOOST if ir_confirmed else IR_PENALTY))

            # bbox format the frontend expects: [x, y, width, height]
            bx = x1; by = y1; bw = x2-x1; bh = y2-y1

            log.info(
                f"[DET] {'CONFIRMED' if ir_confirmed else 'YOLO-only'} | "
                f"fused={fused_conf:.0%} yolo={yolo_conf:.0%} | "
                f"IR {ir_max_temp:.1f}C {ir_hot_px}px | bbox=[{bx},{by},{bw},{bh}]"
            )

            detections_out.append({
                "label":         "person",
                "confidence":    round(fused_conf, 4),
                "bbox":          [bx, by, bw, bh],
                "ir_confirmed":  ir_confirmed,
                "ir_max_temp":   round(ir_max_temp, 1),
                "ir_hot_pixels": ir_hot_px,
            })

        # ── Encode annotated frame as base64 JPEG ────────────────────────────
        annotated = self._annotate(img, detections_out, ir)
        _, jpeg_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")

        # ── Push to all connected frontend clients ────────────────────────────
        self.broadcaster.broadcast_json({
            "type":       "frame",
            "data":       b64,
            "detections": detections_out,
        })

    def _annotate(self, img: np.ndarray, detections: list, ir: Optional[IRFrame]) -> np.ndarray:
        vis = img.copy()

        # IR heatmap thumbnail (top-right corner, 80×80)
        if ir is not None:
            t_min, t_max = ir.temps.min(), ir.temps.max()
            norm    = ((ir.temps - t_min) / max(t_max - t_min, 1e-3) * 255).astype(np.uint8)
            up      = zoom(norm, 10, order=1).astype(np.uint8)
            heatmap = cv2.applyColorMap(up, cv2.COLORMAP_INFERNO)
            h, w    = vis.shape[:2]
            ov_h, ov_w = heatmap.shape[:2]
            xo, yo  = w - ov_w - 4, 4
            roi     = vis[yo:yo+ov_h, xo:xo+ov_w]
            vis[yo:yo+ov_h, xo:xo+ov_w] = cv2.addWeighted(roi, 0.3, heatmap, 0.7, 0)
            cv2.rectangle(vis, (xo, yo), (xo+ov_w, yo+ov_h), (80,80,80), 1)
            cv2.putText(vis, "IR", (xo+2, yo+10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220,220,220), 1)

            # mark warm pixels
            for r in range(8):
                for c in range(8):
                    if IR_HUMAN_MIN_C <= ir.temps[r,c] <= IR_HUMAN_MAX_C:
                        cx = xo + c*10 + 5
                        cy = yo + r*10 + 5
                        cv2.drawMarker(vis, (cx,cy), (0,255,120), cv2.MARKER_CROSS, 5, 1)

        # Bounding boxes
        for d in detections:
            bx, by, bw, bh = d["bbox"]
            col   = (0, 230, 60) if d["ir_confirmed"] else (0, 180, 255)
            label = f"HUMAN {d['confidence']:.0%}" + (" +IR" if d["ir_confirmed"] else "")
            cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), col, 2)
            cv2.putText(vis, label, (bx+2, by-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            if d["ir_confirmed"]:
                cv2.putText(vis, f"{d['ir_max_temp']:.1f}C ({d['ir_hot_pixels']}px)",
                            (bx+2, by+14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100,255,180), 1, cv2.LINE_AA)

        return vis


# ─────────────────────────────────────────────
#  WEBSOCKET SERVER
# ─────────────────────────────────────────────
class WSServer:
    def __init__(self, broadcaster: WSBroadcaster):
        self.broadcaster = broadcaster

    async def handler(self, websocket, path=None):
        # Accept any path — the frontend sends /ws but older clients may vary
        self.broadcaster.register(websocket)
        try:
            async for _ in websocket:
                pass   # we only push, never pull
        except Exception:
            pass
        finally:
            self.broadcaster.unregister(websocket)

    async def run(self):
        log.info(f"[WS ] Server on ws://{WS_HOST}:{WS_PORT}{WS_PATH}")
        async with ws_serve(self.handler, WS_HOST, WS_PORT):
            await asyncio.Future()  # run forever


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    if not _HAS_WEBSOCKETS:
        print("Install websockets:  pip install websockets")
        return

    log.info("=" * 58)
    log.info("  ROVER WebSocket Bridge")
    log.info(f"  UDP  IR :{UDP_PORT_IR}   CAM :{UDP_PORT_CAM}")
    log.info(f"  WS   ws://{WS_HOST}:{WS_PORT}{WS_PATH}")
    log.info(f"  YOLO {YOLO_WEIGHTS}  conf>={YOLO_CONF_THRESH}")
    log.info(f"  IR   {IR_HUMAN_MIN_C}-{IR_HUMAN_MAX_C}°C  boost={IR_BOOST}  penalty={IR_PENALTY}")
    log.info("=" * 58)
    log.info("")
    log.info("  Frontend Settings:")
    log.info(f"    Host:     <this machine's IP>")
    log.info(f"    Port:     {WS_PORT}")
    log.info(f"    Path:     {WS_PATH}")
    log.info(f"    Protocol: ws://")
    log.info("")

    broadcaster = WSBroadcaster()
    pipeline    = Pipeline(broadcaster)

    # Get the asyncio loop before starting threads so broadcast_json can post to it
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    broadcaster.set_loop(loop)

    # Start UDP pipeline on background threads
    pipeline.start()

    # Run the WebSocket server on the asyncio loop (blocks here)
    server = WSServer(broadcaster)
    try:
        loop.run_until_complete(server.run())
    except KeyboardInterrupt:
        log.info("Shutting down.")


if __name__ == "__main__":
    main()