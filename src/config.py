import logging
import os
import time
from dataclasses import dataclass

UDP_HOST = "0.0.0.0"
UDP_PORT_IR = 5005
UDP_PORT_CAM = 5006
WS_HOST = "0.0.0.0"
WS_PORT = 8765
WS_PATH = "/ws"
MAGIC = 0x45533332  # "ES32"
MAGIC_LE = 0x32335345  # "ES32" read as little-endian bytes
BUFFER_SIZE = 65535
SOCKET_RCVBUF = 2 * 1024 * 1024  # 2 MB socket receive buffer

# ── PHONE IP CAMERA CONFIG ───────────────────
# Set this to your phone's camera stream URL.
# Examples:
#   - MJPEG: "http://119.168.1.50:8080/video" (Common for Android 'IP Webcam' app)
#   - RTSP:  "rtsp://192.168.1.50:554/live.sdp"
# Leave as None if you want to fall back to the original ESP32 UDP camera path.
PHONE_STREAM_URL = os.getenv("VIDEO_RECEIVING_IP", "http://172.19.134.12:8080/video")
AUDIO_STREAM_URL = os.getenv("AUDIO_STREAM_URL", "http://172.19.134.12:8080/audio.wav")

# YOLO
YOLO_WEIGHTS = "yolov8n.pt"
YOLO_CONF_THRESH = 0.40
COCO_PERSON_ID = 0
PERSON_NMS_IOU = 0.55  # suppress duplicate YOLO boxes around the same person
TRACK_MATCH_IOU = 0.30  # associate detections with existing tracks across frames
TRACK_MAX_MISSED = 8  # drop a track after this many missed YOLO frames

# IR fusion thresholds
IR_HUMAN_MIN_C = 28.0
IR_HUMAN_MAX_C = 40.0
IR_MIN_HOT_PIXELS = 2
IR_BOOST = 1.30
IR_PENALTY = 0.60
FUSION_MAX_DT_MS = 300

# Pipeline
FRAME_QUEUE_SIZE = 4  # YOLO input queue depth; oldest frame dropped when full
MAX_BROADCAST_FPS = 15  # cap visual frame pushes to frontend
MAX_IR_BROADCAST_FPS = 8  # cap IR grid pushes separately from visual frames
REASSEMBLY_TTL_S = 0.5  # evict stale partial assemblies after this many seconds

# Audio from phone/IP camera streams. FFmpeg must be available on PATH.
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_S = 0.25
AUDIO_ANALYSIS_WINDOW_S = 1.0
AUDIO_ANALYSIS_INTERVAL_S = 1.0
AUDIO_MAX_STALE_MS = 3000
AUDIO_STATUS_INTERVAL_S = 5.0
YAMNET_MODEL_HANDLE = os.getenv(
    "YAMNET_MODEL_HANDLE", "https://tfhub.dev/google/yamnet/1"
)

HUMAN_SOUND_KEYWORDS = (
    "speech",
    "conversation",
    "narration",
    "monologue",
    "babbling",
    "whispering",
    "shout",
    "scream",
    "screaming",
    "yell",
    "crying",
    "baby cry",
    "laughter",
    "chuckle",
    "giggle",
    "snicker",
    "breathing",
    "wheeze",
    "snoring",
    "cough",
    "sneeze",
    "sniff",
    "sigh",
    "groan",
    "grunt",
    "burp",
    "hiccup",
    "heartbeat",
    "hands",
    "clapping",
    "finger snapping",
    "footsteps",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

for handler in logging.getLogger().handlers:
    if handler.formatter is not None:
        handler.formatter.converter = time.localtime
        handler.formatter.default_time_format = "%Y-%m-%d %H:%M:%S"
        handler.formatter.default_msec_format = "%s.%03d"

log = logging.getLogger("robot-backend")


@dataclass(frozen=True)
class AppConfig:
    udp_host: str = UDP_HOST
    udp_port_ir: int = UDP_PORT_IR
    udp_port_cam: int = UDP_PORT_CAM
    ws_host: str = WS_HOST
    ws_port: int = WS_PORT
    ws_path: str = WS_PATH
    phone_stream_url: str | None = PHONE_STREAM_URL
    audio_stream_url: str | None = AUDIO_STREAM_URL


def load_config() -> AppConfig:
    return AppConfig()
