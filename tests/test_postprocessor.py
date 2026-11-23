import sys
from pathlib import Path

import pytest


SRC = Path(__file__).parents[1] / "02_CODE/src"
sys.path.insert(0, str(SRC))

from tb_afb.inference.postprocessor import DetectionPostprocessor


def detection(x, y, width, height, confidence=0.9):
    return {
        "bbox": [x, y, width, height],
        "confidence": confidence,
        "class_id": 0,
    }


def test_morphology_filter_requires_calibration():
    processor = DetectionPostprocessor(enable_morphology_filter=True)
    with pytest.raises(ValueError, match="microns-per-pixel"):
        processor.filter([detection(10, 10, 8, 2)])


def test_morphology_filter_is_opt_in():
    processor = DetectionPostprocessor(enable_morphology_filter=False)
    result = processor.filter([detection(100, 100, 400, 400)])
    assert len(result) == 1


def test_global_nms_removes_overlapping_duplicate():
    processor = DetectionPostprocessor(
        min_confidence=0.25, nms_iou_threshold=0.5
    )
    result = processor.filter(
        [
            detection(100, 100, 20, 10, 0.9),
            detection(101, 100, 20, 10, 0.8),
        ]
    )
    assert len(result) == 1
    assert result[0]["confidence"] == 0.9



@pytest.mark.parametrize(
    ("bbox", "confidence"),
    [
        ([float("nan"), 10, 8, 2], 0.9),
        ([10, 10, float("inf"), 2], 0.9),
        ([10, 10, 8, 2], float("nan")),
        ([10, 10, 8, -2], 0.9),
        ([10, 10, 8], 0.9),
        (["invalid", 10, 8, 2], 0.9),
    ],
)
def test_invalid_detections_are_excluded(bbox, confidence):
    processor = DetectionPostprocessor(min_confidence=0.25)
    assert processor.filter([{"bbox": bbox, "confidence": confidence}]) == []


def test_morphology_filter_rejects_nonfinite_calibration():
    processor = DetectionPostprocessor(enable_morphology_filter=True)
    with pytest.raises(ValueError, match="microns-per-pixel"):
        processor.filter([detection(10, 10, 8, 2)], pixel_size_microns=float("nan"))


def test_thresholds_must_be_valid():
    with pytest.raises(ValueError, match="min_confidence"):
        DetectionPostprocessor(min_confidence=float("nan"))
    with pytest.raises(ValueError, match="nms_iou_threshold"):
        DetectionPostprocessor(nms_iou_threshold=1.5)
