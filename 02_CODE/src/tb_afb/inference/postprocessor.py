from typing import Dict, List

import numpy as np


class DetectionPostprocessor:
    """Filter candidate boxes by confidence, optional morphology, and global NMS."""

    MIN_ASPECT_RATIO = 2.0
    MAX_ASPECT_RATIO = 10.0
    MIN_AREA_MICRONS = 2.0
    MAX_AREA_MICRONS = 20.0

    def __init__(
        self,
        min_confidence: float = 0.3,
        nms_iou_threshold: float = 0.5,
        *,
        enable_morphology_filter: bool = False,
    ):
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1].")
        if not 0.0 <= nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be in [0, 1].")
        self.min_confidence = min_confidence
        self.nms_iou_threshold = nms_iou_threshold
        self.enable_morphology_filter = enable_morphology_filter

    def filter(
        self,
        detections: List[Dict],
        *,
        pixel_size_microns: float | None = None,
    ) -> List[Dict]:
        if not detections:
            return []
        if self.enable_morphology_filter and (
            pixel_size_microns is None
            or not np.isfinite(pixel_size_microns)
            or pixel_size_microns <= 0
        ):
            raise ValueError(
                "Morphology filtering requires a positive calibrated microns-per-pixel value."
            )

        filtered = []
        boxes = []
        scores = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            bbox = detection.get("bbox")
            if not isinstance(bbox, (list, tuple, np.ndarray)) or len(bbox) != 4:
                continue
            try:
                cx, cy, width, height = (float(value) for value in bbox)
                confidence = float(detection.get("confidence", 0))
            except (TypeError, ValueError, OverflowError):
                continue
            if not np.isfinite([cx, cy, width, height, confidence]).all():
                continue
            if width <= 0 or height <= 0 or confidence < self.min_confidence:
                continue

            if self.enable_morphology_filter:
                aspect_ratio = max(width, height) / min(width, height)
                area_microns = width * height * pixel_size_microns ** 2
                if not self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO:
                    continue
                if not self.MIN_AREA_MICRONS <= area_microns <= self.MAX_AREA_MICRONS:
                    continue

            filtered.append(detection)
            boxes.append([
                cx - width / 2,
                cy - height / 2,
                cx + width / 2,
                cy + height / 2,
            ])
            scores.append(confidence)

        if not filtered:
            return []
        keep_indices = self.apply_nms(
            np.asarray(boxes, dtype=np.float64),
            np.asarray(scores, dtype=np.float64),
            self.nms_iou_threshold,
        )
        return [filtered[i] for i in keep_indices]

    @staticmethod
    def apply_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float = 0.5,
    ) -> List[int]:
        boxes = np.asarray(boxes, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes must be an (N, 4) array.")
        if scores.ndim != 1 or len(scores) != len(boxes):
            raise ValueError("scores must contain one value per box.")
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError("boxes and scores must be finite.")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1].")
        if len(boxes) == 0:
            return []

        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size:
            current = int(order[0])
            keep.append(current)
            remaining = order[1:]
            if remaining.size == 0:
                break

            xx1 = np.maximum(x1[current], x1[remaining])
            yy1 = np.maximum(y1[current], y1[remaining])
            xx2 = np.minimum(x2[current], x2[remaining])
            yy2 = np.minimum(y2[current], y2[remaining])
            intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            union = areas[current] + areas[remaining] - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
            order = remaining[iou <= iou_threshold]

        return keep
