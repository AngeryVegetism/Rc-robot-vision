from pydantic import BaseModel

# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class SignalPacket(BaseModel):
    """Payload sent by Arduino over WiFi for non-video signals."""
    ir: float | None = None          # PIR / IR proximity value
    acoustic: float | None = None    # Microphone amplitude (RMS)
    radar: float | None = None       # Micro-Doppler magnitude
    co2: float | None = None         # CO₂ ppm
    thermal: float | None = None     # Thermal array mean temperature (°C)
    vibration: float | None = None   # Accelerometer magnitude
    timestamp: float | None = None


class DiagnosisResponse(BaseModel):
    verdict: str          # "ALIVE" | "NOT_ALIVE" | "UNKNOWN"
    confidence: float     # 0.0 – 1.0
    signals: dict
    detections: list
    timestamp: float