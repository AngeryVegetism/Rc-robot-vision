import time
from typing import Dict, List, Set, Tuple

import numpy as np

from .config import (
    COCO_PERSON_ID,
    IR_HUMAN_MAX_C,
    IR_HUMAN_MIN_C,
    IR_MIN_HOT_PIXELS,
    PERSON_NMS_IOU,
    TRACK_MATCH_IOU,
    TRACK_MAX_MISSED,
    YOLO_CONF_THRESH,
    YOLO_WEIGHTS,
    log,
)
from .models import IRFrame


class YOLODetector:
    def __init__(self):
        try:
            from ultralytics import YOLO

            log.info(f"[YOLO] Loading {YOLO_WEIGHTS}")
            self._model = YOLO(YOLO_WEIGHTS)
            self._backend = "ultralytics"
            log.info("[YOLO] Ready")
        except ImportError:
            log.warning("[YOLO] ultralytics not found — detection disabled")
            self._model = None
            self._backend = "none"

    def detect_persons(
        self, img: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        if self._backend != "ultralytics":
            return []
        results = self._model(
            img, conf=YOLO_CONF_THRESH, classes=[COCO_PERSON_ID], verbose=False
        )
        out = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                out.append(((x1, y1, x2, y2), conf))
        return out


# ─────────────────────────────────────────────
#  IR ANALYSER
# ─────────────────────────────────────────────
def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def suppress_duplicate_persons(
    detections: List[Tuple[Tuple[int, int, int, int], float]],
    iou_thresh: float = PERSON_NMS_IOU,
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    kept: List[Tuple[Tuple[int, int, int, int], float]] = []
    for bbox, conf in sorted(detections, key=lambda item: item[1], reverse=True):
        if all(bbox_iou(bbox, kept_bbox) < iou_thresh for kept_bbox, _ in kept):
            kept.append((bbox, conf))
    return kept


class PersonTracker:
    def __init__(
        self, match_iou: float = TRACK_MATCH_IOU, max_missed: int = TRACK_MAX_MISSED
    ):
        self._match_iou = match_iou
        self._max_missed = max_missed
        self._next_id = 1
        self._tracks: Dict[int, dict] = {}

    def update(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], float]],
    ) -> List[Tuple[Tuple[int, int, int, int], float, int]]:
        assigned_track_ids: Set[int] = set()
        tracked: List[Tuple[Tuple[int, int, int, int], float, int]] = []

        for bbox, conf in sorted(detections, key=lambda item: item[1], reverse=True):
            best_id = None
            best_iou = self._match_iou

            for track_id, track in self._tracks.items():
                if track_id in assigned_track_ids:
                    continue
                iou = bbox_iou(bbox, track["bbox"])
                if iou >= best_iou:
                    best_iou = iou
                    best_id = track_id

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1

            self._tracks[best_id] = {
                "bbox": bbox,
                "missed": 0,
                "last_seen": time.monotonic(),
            }
            assigned_track_ids.add(best_id)
            tracked.append((bbox, conf, best_id))

        stale_ids = []
        for track_id, track in self._tracks.items():
            if track_id in assigned_track_ids:
                continue
            track["missed"] += 1
            if track["missed"] > self._max_missed:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            del self._tracks[track_id]

        return tracked


class IRAnalyser:
    def analyse(self, bbox: Tuple, img_shape: Tuple, ir: IRFrame):
        img_h, img_w = img_shape
        x1, y1, x2, y2 = [max(0, v) for v in bbox]
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        c1 = max(0, int(x1 / img_w * 8))
        c2 = min(7, int(x2 / img_w * 8))
        r1 = max(0, int(y1 / img_h * 8))
        r2 = min(7, int(y2 / img_h * 8))
        if c1 == c2:
            c2 = min(7, c2 + 1)
        if r1 == r2:
            r2 = min(7, r2 + 1)

        region = ir.temps[r1 : r2 + 1, c1 : c2 + 1]
        if region.size == 0:
            return False, 0.0, 0

        max_t = float(region.max())
        hot_px = int(np.sum((region >= IR_HUMAN_MIN_C) & (region <= IR_HUMAN_MAX_C)))
        return hot_px >= IR_MIN_HOT_PIXELS, max_t, hot_px
