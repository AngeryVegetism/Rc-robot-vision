"""
Search & Rescue — Multi-Modal Vitals Detection System
FastAPI Backend: Video + IR + Acoustic + Radar signal processing
with YOLO object detection and alive/not-alive classification.
"""

import asyncio
import json
import time
import numpy as np
from collections import deque
from typing import AsyncGenerator

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from tools.schemas import SignalPacket, DiagnosisResponse
from tools.signalprocessing import bandpass_filter, estimate_breathing_rate, estimate_heart_rate, signal_energy, SCIPY_AVAILABLE

# ── Optional YOLO import (graceful fallback if ultralytics not installed) ──────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed — YOLO detections will be simulated.")

# ── Optional audio/signal libs ────────────────────────────────────────────────


app = FastAPI(
    title="Search & Rescue Vitals API",
    description="Real-time multi-modal alive detection using YOLO + signal processing",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════════

# Rolling buffers (last 200 samples per signal)
BUFFER_SIZE = 200
signal_buffers: dict[str, deque] = {
    "ir":        deque(maxlen=BUFFER_SIZE),
    "acoustic":  deque(maxlen=BUFFER_SIZE),
    "radar":     deque(maxlen=BUFFER_SIZE),
    "co2":       deque(maxlen=BUFFER_SIZE),
    "thermal":   deque(maxlen=BUFFER_SIZE),
    "vibration": deque(maxlen=BUFFER_SIZE),
}

# Latest diagnosis
latest_diagnosis: dict = {
    "verdict": "UNKNOWN",
    "confidence": 0.0,
    "signals": {},
    "detections": [],
    "timestamp": time.time(),
}

# Connected WebSocket clients
ws_clients: list[WebSocket] = []

# YOLO model singleton
_yolo_model = None


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None and YOLO_AVAILABLE:
        # Uses YOLOv8n by default; swap for custom trained weights as needed
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


# ═══════════════════════════════════════════════════════════════════════════════
# ALIVE / NOT-ALIVE FUSION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_diagnosis(detections: list[dict]) -> dict:
    """
    Multi-modal fusion decision engine.

    Scoring weights:
      - IR motion present          → +20 pts
      - Acoustic energy > threshold → +15 pts
      - Breathing rate 8–30/min    → +25 pts
      - Heart rate 40–180 BPM      → +25 pts
      - CO₂ elevated (>450 ppm)    → +15 pts
      - Thermal temp 30–40°C       → +20 pts
      - Vibration detected         → +10 pts
      - YOLO person detected       → +20 pts
    Max = 150 pts → normalised to confidence 0–1
    """
    score = 0
    max_score = 150
    signal_summary = {}

    ir_data       = list(signal_buffers["ir"])
    acoustic_data = list(signal_buffers["acoustic"])
    radar_data    = list(signal_buffers["radar"])
    co2_data      = list(signal_buffers["co2"])
    thermal_data  = list(signal_buffers["thermal"])
    vib_data      = list(signal_buffers["vibration"])

    # IR motion
    if ir_data:
        ir_mean = float(np.mean(ir_data[-20:]))
        ir_motion = ir_mean > 0.3
        signal_summary["ir_motion"] = ir_motion
        signal_summary["ir_value"] = round(ir_mean, 3)
        if ir_motion:
            score += 20

    # Acoustic energy
    acoustic_rms = signal_energy(acoustic_data[-50:])
    signal_summary["acoustic_rms"] = round(acoustic_rms, 4)
    if acoustic_rms > 0.05:
        score += 15

    # Breathing rate
    breath_rate = estimate_breathing_rate(acoustic_data)
    signal_summary["breathing_rate_bpm"] = breath_rate
    if 8 <= breath_rate <= 30:
        score += 25

    # Heart rate
    heart_rate = estimate_heart_rate(radar_data)
    signal_summary["heart_rate_bpm"] = heart_rate
    if 40 <= heart_rate <= 180:
        score += 25

    # CO₂
    if co2_data:
        co2_mean = float(np.mean(co2_data[-20:]))
        signal_summary["co2_ppm"] = round(co2_mean, 1)
        if co2_mean > 450:
            score += 15

    # Thermal
    if thermal_data:
        temp_mean = float(np.mean(thermal_data[-20:]))
        signal_summary["thermal_temp_c"] = round(temp_mean, 1)
        if 30 <= temp_mean <= 40:
            score += 20

    # Vibration
    vib_rms = signal_energy(vib_data[-20:])
    signal_summary["vibration_rms"] = round(vib_rms, 4)
    if vib_rms > 0.02:
        score += 10

    # YOLO person detected in current frame
    person_detected = any(d.get("label") == "person" for d in detections)
    signal_summary["person_in_frame"] = person_detected
    if person_detected:
        score += 20

    confidence = min(score / max_score, 1.0)

    if confidence >= 0.55:
        verdict = "ALIVE"
    elif confidence >= 0.25:
        verdict = "UNCERTAIN"
    else:
        verdict = "NOT_ALIVE"

    return {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "score": score,
        "max_score": max_score,
        "signals": signal_summary,
        "detections": detections,
        "timestamp": time.time(),
    }



# ═══════════════════════════════════════════════════════════════════════════════
# YOLO INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def run_yolo(frame_bytes: bytes) -> list[dict]:
    """Run YOLO on a JPEG frame. Returns list of detection dicts."""
    model = get_yolo_model()
    if model is None:
        # Simulated detections for dev/testing
        return [{"label": "person", "confidence": 0.87, "bbox": [100, 80, 320, 460]}]

    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return []

    results = model(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        label = model.names[int(box.cls[0])]
        conf  = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "label": label,
            "confidence": round(conf, 3),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })
    return detections


def annotate_frame(frame_bytes: bytes, detections: list[dict],
                   verdict: str, confidence: float) -> bytes:
    """Draw bounding boxes + verdict overlay on frame, return JPEG bytes."""
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return frame_bytes

    # Verdict overlay
    color_map = {"ALIVE": (0, 255, 80), "NOT_ALIVE": (0, 60, 255), "UNCERTAIN": (0, 200, 255), "UNKNOWN": (180, 180, 180)}
    color = color_map.get(verdict, (180, 180, 180))
    cv2.rectangle(frame, (0, 0), (360, 48), (10, 10, 10), -1)
    cv2.putText(frame, f"{verdict}  {confidence*100:.0f}%", (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = f"{d['label']} {d['confidence']*100:.0f}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


# ═══════════════════════════════════════════════════════════════════════════════
# BROADCAST HELPER
# ═══════════════════════════════════════════════════════════════════════════════

async def broadcast(payload: dict):
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_clients.remove(ws)


# ═══════════════════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {"status": "ok", "service": "Search & Rescue Vitals API"}


@app.get("/health")
async def health():
    return {
        "yolo_available": YOLO_AVAILABLE,
        "scipy_available": SCIPY_AVAILABLE,
        "buffer_sizes": {k: len(v) for k, v in signal_buffers.items()},
    }


@app.post("/stream/signals", summary="Ingest sensor packet from Arduino")
async def ingest_signals(packet: SignalPacket):
    """
    Receive a JSON signal packet from the Arduino WiFi module.
    Appends values to rolling buffers and recomputes diagnosis.
    """
    ts = packet.timestamp or time.time()
    mapping = {
        "ir":        packet.ir,
        "acoustic":  packet.acoustic,
        "radar":     packet.radar,
        "co2":       packet.co2,
        "thermal":   packet.thermal,
        "vibration": packet.vibration,
    }
    for key, val in mapping.items():
        if val is not None:
            signal_buffers[key].append(val)

    # Re-run diagnosis with existing detections
    diagnosis = run_diagnosis(latest_diagnosis.get("detections", []))
    latest_diagnosis.update(diagnosis)

    await broadcast({"type": "diagnosis", **diagnosis})
    await broadcast({"type": "signals", "data": {
        k: list(v)[-50:] for k, v in signal_buffers.items()
    }, "timestamp": ts})

    return {"status": "ok", "verdict": diagnosis["verdict"]}


@app.post("/stream/video", summary="Ingest single JPEG frame from camera")
async def ingest_video_frame(file: UploadFile = File(...)):
    """
    Receive a JPEG frame from the Arduino/ESP32-CAM.
    Runs YOLO detection, updates diagnosis, broadcasts annotated frame.
    """
    frame_bytes = await file.read()
    detections  = run_yolo(frame_bytes)
    diagnosis   = run_diagnosis(detections)
    latest_diagnosis.update(diagnosis)

    annotated = annotate_frame(
        frame_bytes, detections,
        diagnosis["verdict"], diagnosis["confidence"]
    )

    import base64
    await broadcast({
        "type": "video_frame",
        "frame_b64": base64.b64encode(annotated).decode(),
        "detections": detections,
        "verdict": diagnosis["verdict"],
        "confidence": diagnosis["confidence"],
        "timestamp": time.time(),
    })

    return {"status": "ok", "detections": detections, "verdict": diagnosis["verdict"]}


@app.get("/diagnosis", response_model=DiagnosisResponse,
         summary="Get latest diagnosis snapshot")
async def get_diagnosis():
    return latest_diagnosis


@app.get("/signals/history", summary="Get rolling signal buffers")
async def get_signal_history():
    return {k: list(v) for k, v in signal_buffers.items()}


@app.delete("/signals/reset", summary="Clear all signal buffers")
async def reset_signals():
    for buf in signal_buffers.values():
        buf.clear()
    latest_diagnosis.update({"verdict": "UNKNOWN", "confidence": 0.0,
                              "signals": {}, "detections": [], "timestamp": time.time()})
    return {"status": "reset"}


# ═══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET — real-time push to dashboard
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Dashboard connects here. Receives:
      - { type: "diagnosis", verdict, confidence, signals, detections, timestamp }
      - { type: "signals",   data: { ir:[...], acoustic:[...], ... }, timestamp }
      - { type: "video_frame", frame_b64, detections, verdict, confidence, timestamp }
    """
    await websocket.accept()
    ws_clients.append(websocket)
    try:
        # Send current state immediately on connect
        await websocket.send_text(json.dumps({
            "type": "init",
            "diagnosis": latest_diagnosis,
            "signals": {k: list(v)[-50:] for k, v in signal_buffers.items()},
        }))
        while True:
            # Keep-alive: echo pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        if websocket in ws_clients:
            ws_clients.remove(websocket)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO DATA GENERATOR  (GET /demo/start  — for testing without hardware)
# ═══════════════════════════════════════════════════════════════════════════════

_demo_task: asyncio.Task | None = None

@app.get("/demo/start", summary="Start synthetic data feed for dashboard testing")
async def demo_start():
    global _demo_task
    if _demo_task and not _demo_task.done():
        return {"status": "already running"}

    async def _generate():
        t = 0.0
        while True:
            t += 0.1
            packet = SignalPacket(
                ir        = abs(np.sin(t * 0.8) + np.random.normal(0, 0.05)),
                acoustic  = abs(np.sin(t * 0.25) * 0.4 + np.random.normal(0, 0.03)),
                radar     = abs(np.sin(t * 1.4) * 0.6 + np.random.normal(0, 0.05)),
                co2       = 450 + 30 * np.sin(t * 0.1) + np.random.normal(0, 5),
                thermal   = 36.5 + np.sin(t * 0.05) + np.random.normal(0, 0.2),
                vibration = abs(np.random.normal(0, 0.03) + 0.01 * np.sin(t * 2)),
                timestamp = time.time(),
            )
            await ingest_signals(packet)
            await asyncio.sleep(0.1)

    _demo_task = asyncio.create_task(_generate())
    return {"status": "demo started"}


@app.get("/demo/stop", summary="Stop synthetic data feed")
async def demo_stop():
    global _demo_task
    if _demo_task:
        _demo_task.cancel()
        _demo_task = None
    return {"status": "demo stopped"}


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)