from typing import List, Optional, Tuple

import numpy as np

from .config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_STREAM_URL,
    HUMAN_SOUND_KEYWORDS,
    YAMNET_MODEL_HANDLE,
    log,
)
from .models import AudioState


class YAMNetAnalyser:
    def __init__(self):
        self._model = None
        self._class_names: List[str] = []
        self._human_indices: List[int] = []

        try:
            import tensorflow_hub as hub

            log.info(f"[YAMNet] Loading {YAMNET_MODEL_HANDLE}")
            self._model = hub.load(YAMNET_MODEL_HANDLE)
            self._class_names = self._load_class_names()
            self._human_indices = [
                i
                for i, name in enumerate(self._class_names)
                if any(keyword in name.lower() for keyword in HUMAN_SOUND_KEYWORDS)
            ]
            log.info(
                f"[YAMNet] Ready ({len(self._human_indices)} human-related classes)"
            )
        except Exception as e:
            log.warning(f"[YAMNet] unavailable — audio classification disabled: {e}")
            self._model = None

    def _load_class_names(self) -> List[str]:
        try:
            import csv

            class_map_path = self._model.class_map_path().numpy().decode("utf-8")
            with open(class_map_path, newline="", encoding="utf-8") as f:
                return [row["display_name"] for row in csv.DictReader(f)]
        except Exception as e:
            log.warning(f"[YAMNet] class map unavailable: {e}")
            return []

    def analyse(self, waveform: np.ndarray) -> AudioState:
        db = audio_db(waveform)
        if self._model is None or not self._human_indices:
            return AudioState(db=db, confidence=0.0, labels=[])

        try:
            scores, _, _ = self._model(waveform.astype(np.float32))
            mean_scores = np.asarray(scores).mean(axis=0)
        except Exception as e:
            log.warning(f"[YAMNet] inference failed: {e}")
            return AudioState(db=db, confidence=0.0, labels=[])

        ranked = sorted(
            ((idx, float(mean_scores[idx])) for idx in self._human_indices),
            key=lambda item: item[1],
            reverse=True,
        )
        top = [
            (self._class_names[idx], score)
            for idx, score in ranked[:3]
            if score >= 0.10
        ]
        confidence = max((score for _, score in top), default=0.0)
        return AudioState(
            db=db,
            confidence=confidence,
            labels=[f"{label} {score:.0%}" for label, score in top],
        )


def audio_db(waveform: np.ndarray) -> float:
    if waveform.size == 0:
        return -100.0
    rms = float(np.sqrt(np.mean(np.square(waveform))))
    return 20.0 * np.log10(max(rms, 1e-5))


def combine_life_signs(
    yolo_conf: float, ir_confirmed: bool, audio: Optional[AudioState]
) -> Tuple[float, str]:

    thermal = 1.0 if ir_confirmed else 0.0
    audio_conf = audio.confidence if audio and audio.human_related else 0.0

    score = 0.40 * yolo_conf + 0.35 * thermal + 0.25 * audio_conf

    # Boost confidence when multiple modalities agree
    modalities = sum([yolo_conf > 0.5, ir_confirmed, audio_conf > 0.5])

    if modalities >= 2:
        score = min(1.0, score + 0.15)

    if score >= 0.80:
        status = "confirmed_life_signs"
    elif score >= 0.60:
        status = "likely_alive"
    elif score >= 0.35:
        status = "possible_life_signs"
    else:
        status = "inconclusive"

    return score, status


def build_audio_ffmpeg_cmd(url: Optional[str] = None) -> List[str]:
    return [
        "ffmpeg",
        "-loglevel",
        "error",
        "-f",
        "wav",
        "-i",
        url or AUDIO_STREAM_URL,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-f",
        "s16le",
        "pipe:1",
    ]
