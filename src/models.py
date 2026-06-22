import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class IRFrame:
    seq: int
    timestamp: int
    temps: np.ndarray  # (8, 8) float32  °C
    received: float = field(default_factory=time.monotonic)


@dataclass
class CamFrame:
    seq: int
    timestamp: int
    jpeg: bytes
    received: float = field(default_factory=time.monotonic)
    raw_img: Optional[np.ndarray] = None  # Added back-compat for stream raw decodes


@dataclass
class AudioState:
    db: float
    confidence: float
    labels: List[str]
    received: float = field(default_factory=time.monotonic)

    @property
    def human_related(self) -> bool:
        return self.confidence > 0.0
