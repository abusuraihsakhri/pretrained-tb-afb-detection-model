"""Lightweight tests for the architecture comparison scientific safety gate."""
import csv
import struct
import sys
import zlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.compare_architectures import (  # noqa: E402
    ARCHITECTURES,
    digest,
    read_dataset,
    scientific_gate,
)
from prepare_curated_dataset import FIELDS  # noqa: E402


def make_png(rgb):
    """Small valid RGB PNG, without image libraries or network dependencies."""
    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00" + bytes(rgb)))
            + chunk(b"IEND", b""))


def fixture(tmp_path):
    original = tmp_path / "original"
    curated = tmp_path / "curated"
    rows = []
    for i, split in enumerate(("train", "val", "test")):
        name = f"case_{split}"
        for root in (original, curated):
            images = root / split / "images"
            labels = root / split / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (images / f"{name}.png").write_bytes(make_png((20 + i, 150, 200)))
            (labels / f"{name}.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
            )
        image = original / split / "images" / f"{name}.png"
        rows.append({
            "record_id": name, "source_id": "external" if split == "test" else "internal",
            "source_version": "v1", "original_split": split,
            "image_path": str(image.relative_to(original)),
            "label_path": f"{split}/labels/{name}.txt",
            "content_sha256": digest(image),
            "biological_group_id": f"slide-{split}",
            "target_split": split, "license_id": "CC-BY-4.0",
            "provenance_status": "verified", "include": "true",
            "exclusion_reason": "",
        })

    manifest = tmp_path / "reviewed.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "nc": 1, "names": {0: "AFB"}, "path": str(curated),
        "train": "train/images", "val": "val/images", "test": "test/images",
    }), encoding="utf-8")
    return original, curated, manifest, data_yaml, rows


def rewrite_manifest(manifest, rows):
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_supported_architectures():
    assert set(ARCHITECTURES) == {
        "yolov8n", "yolov8s", "yolov8s_p2", "rtdetr_l"
    }


def test_gate_passes_curated_reviewed_data(tmp_path):
    original, _, manifest, yaml_path, _ = fixture(tmp_path)
    result = scientific_gate(yaml_path, manifest, original, {"external"})
    assert result["integrity_audit"]["passed"] is True
    assert result["reviewed_rows"] == 3
    assert result["integrity_audit"]["total_images"] == 3


def test_requires_source_holdout(tmp_path):
    original, _, manifest, yaml_path, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="test-source"):
        scientific_gate(yaml_path, manifest, original, set())


def test_rejects_biological_group_leakage(tmp_path):
    original, _, manifest, yaml_path, rows = fixture(tmp_path)
    rows[1]["biological_group_id"] = rows[0]["biological_group_id"]
    rewrite_manifest(manifest, rows)
    with pytest.raises(ValueError, match="Biological groups cross splits"):
        scientific_gate(yaml_path, manifest, original, {"external"})


def test_rejects_image_leakage(tmp_path):
    original, curated, manifest, yaml_path, rows = fixture(tmp_path)
    duplicate = (original / "train/images/case_train.png").read_bytes()
    for root in (original, curated):
        (root / "val/images/case_val.png").write_bytes(duplicate)
    rows[1]["content_sha256"] = digest(original / "val/images/case_val.png")
    rewrite_manifest(manifest, rows)
    with pytest.raises(ValueError, match="Exact duplicate content crosses splits"):
        scientific_gate(yaml_path, manifest, original, {"external"})


def test_rejects_edited_annotation(tmp_path):
    original, curated, manifest, yaml_path, _ = fixture(tmp_path)
    (curated / "val/labels/case_val.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Curated annotation differs"):
        scientific_gate(yaml_path, manifest, original, {"external"})


def test_rejects_bad_annotations(tmp_path):
    original, curated, manifest, yaml_path, _ = fixture(tmp_path)
    for root in (original, curated):
        (root / "val/labels/case_val.txt").write_text(
            "0 0.9 0.9 0.8 0.8\n", encoding="utf-8"
        )
    with pytest.raises(ValueError, match="integrity audit failed"):
        scientific_gate(yaml_path, manifest, original, {"external"})


def test_rejects_bad_class_mapping(tmp_path):
    _, _, _, data_yaml, _ = fixture(tmp_path)
    config = yaml.safe_load(data_yaml.read_text())
    config["names"] = {0: "artifact"}
    data_yaml.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="one annotated class"):
        read_dataset(data_yaml)
