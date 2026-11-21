#!/usr/bin/env python3
"""Scientifically gated AFB architecture comparison.

No training occurs without a verified provenance manifest, biological grouping,
a whole-source held-out test split, and a passing all-split data integrity audit.
This script NEVER evaluates the locked test split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_CODE" / "scripts"))
from check_data_integrity import audit_dataset  # noqa: E402
from prepare_curated_dataset import (  # noqa: E402
    load_reviewed_manifest,
    validate_manifest,
)

ARCHITECTURES = {
    "yolov8n": {"weights": "yolov8n.pt", "kind": "cnn", "yaml": None},
    "yolov8s": {"weights": "yolov8s.pt", "kind": "cnn", "yaml": None},
    "yolov8s_p2": {
        "weights": "yolov8s.pt",
        "kind": "cnn_p2",
        "yaml": "yolov8s-p2.yaml",
    },
    "rtdetr_l": {"weights": "rtdetr-l.pt", "kind": "hybrid_transformer", "yaml": None},
}
SPLITS = ("train", "val", "test")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_dataset(data_yaml: Path) -> Path:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Dataset YAML is not a mapping")
    names = config.get("names")
    valid_names = names in (["AFB"], {0: "AFB"}, {"0": "AFB"})
    if config.get("nc") != 1 or not valid_names:
        raise ValueError("Expected exactly one annotated class: AFB")
    if any(config.get(key) != f"{split}/images" for key, split in
           (("train", "train"), ("val", "val"), ("test", "test"))):
        raise ValueError("Expected explicit train/val/test image directories")
    if not config.get("path"):
        raise ValueError("Missing dataset path")
    root = Path(str(config["path"])).expanduser()
    return (root if root.is_absolute() else data_yaml.parent / root).resolve()


def check_materialized_rows(rows: list[dict[str, str]],
                            source_root: Path, curated_root: Path) -> None:
    """Compare every reviewed image/label to the materialized dataset."""
    expected = {split: set() for split in SPLITS}
    for row in rows:
        split = row["target_split"].strip().lower()
        src_img = (source_root / row["image_path"]).resolve()
        src_lab = (source_root / row["label_path"]).resolve()
        dest_img = curated_root / split / "images" / src_img.name
        dest_lab = curated_root / split / "labels" / src_lab.name
        if not dest_img.is_file() or not dest_lab.is_file():
            raise ValueError(f"Curated copy missing: {split}/{src_img.name}")
        if digest(dest_img) != row["content_sha256"].strip().lower():
            raise ValueError(f"Curated image differs from reviewed manifest: {src_img.name}")
        if digest(dest_lab) != digest(src_lab):
            raise ValueError(f"Curated annotation differs from reviewed source: {src_lab.name}")
        if src_img.name in expected[split]:
            raise ValueError(f"Duplicate output filename in {split}: {src_img.name}")
        expected[split].add(src_img.name)

    for split in SPLITS:
        image_dir = curated_root / split / "images"
        image_names = {p.name for p in image_dir.iterdir() if p.is_file()}
        if image_names != expected[split]:
            extras = sorted(image_names - expected[split])[:5]
            missing = sorted(expected[split] - image_names)[:5]
            raise ValueError(f"Curated {split} differs from reviewed manifest: "
                             f"extra={extras}, missing={missing}")
        if not expected[split]:
            raise ValueError(f"Curated {split} is empty")


def scientific_gate(data_yaml: Path, manifest: Path,
                    source_root: Path, test_sources: set[str]) -> dict:
    curated_root = read_dataset(data_yaml)
    if not source_root.is_dir() or not curated_root.is_dir():
        raise ValueError("Original and curated datasets must both exist locally")
    rows = load_reviewed_manifest(manifest)
    selected = validate_manifest(rows, source_root, test_sources)
    check_materialized_rows(selected, source_root, curated_root)
    report = audit_dataset(curated_root)
    if not report.passed:
        details = "; ".join(report.errors[:4])
        raise ValueError(f"Dataset integrity audit failed: {details}")
    return {
        "source_root": str(source_root),
        "dataset_root": str(curated_root),
        "reviewed_manifest_sha256": digest(manifest),
        "data_yaml_sha256": digest(data_yaml),
        "test_sources_held_out": sorted(test_sources),
        "integrity_audit": report.to_dict(),
        "reviewed_rows": len(selected),
    }


def load_architecture(model_id: str, weight: Path):
    """Load only explicitly supplied local weights; never fetch silently."""
    if not weight.is_file():
        raise FileNotFoundError(
            f"Missing {weight}; acquire and verify official pretrained weights first"
        )
    from ultralytics import RTDETR, YOLO

    if model_id == "yolov8s_p2":
        model = YOLO(ARCHITECTURES[model_id]["yaml"])
        model.load(str(weight))
        strides = {int(x) for x in model.model.stride.tolist()}
        if strides != {4, 8, 16, 32}:
            raise RuntimeError(f"P2 feature head absent; found strides {sorted(strides)}")
    elif model_id == "rtdetr_l":
        model = RTDETR(str(weight))
    else:
        model = YOLO(str(weight))
    return model


def git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--reviewed-manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--test-source", action="append")
    parser.add_argument("--model", action="append", choices=sorted(ARCHITECTURES))
    parser.add_argument("--weights-dir", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "06_LOGS" / "architecture_comparison")
    parser.add_argument("--train", action="store_true",
                        help="Actually train after all data and weight checks")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_models:
        print(json.dumps(ARCHITECTURES, indent=2))
        return 0
    required = (args.data, args.reviewed_manifest, args.source_root, args.test_source)
    if not all(required):
        raise SystemExit(
            "Required: --data --reviewed-manifest --source-root --test-source. "
            "Start by curating and reviewing dataset provenance."
        )
    if min(args.epochs, args.batch, args.imgsz) < 1:
        raise SystemExit("Epochs, batch and input size must be positive")
    data_yaml = args.data.expanduser().resolve()
    manifest = args.reviewed_manifest.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    check = scientific_gate(data_yaml, manifest, source_root, set(args.test_source))
    print(json.dumps({
        "data_gate": "PASS", "images": check["integrity_audit"]["total_images"],
        "reviewed_rows": check["reviewed_rows"],
        "test_source_holdout": check["test_sources_held_out"],
    }, indent=2))
    if not args.train:
        print("PRE-FLIGHT ONLY: no model trained and no test data evaluated.")
        return 0

    # Ensure actual image decoding occurs during training gate.
    import cv2  # noqa: F401
    selected = args.model or list(ARCHITECTURES)
    weights_dir = args.weights_dir.expanduser().resolve()
    weights = {name: weights_dir / ARCHITECTURES[name]["weights"]
               for name in selected}
    missing = sorted(str(p) for p in weights.values() if not p.is_file())
    if missing:
        raise SystemExit("Missing local pretrained weight files: " + ", ".join(missing))

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    project_dir = args.output.expanduser().resolve() / run_stamp
    for name in selected:
        spec = ARCHITECTURES[name]
        model = load_architecture(name, weights[name])
        is_transformer = spec["kind"] == "hybrid_transformer"
        model.train(
            data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, device=args.device, seed=args.seed,
            deterministic=not is_transformer, workers=0,
            project=str(project_dir), name=name, exist_ok=False,
            plots=True, save=True,
        )
        run_dir = Path(model.trainer.save_dir)
        run_info = {
            "status": "development_only_not_independent_validation",
            "model": name, "architecture": spec["kind"],
            "base_weights_sha256": digest(weights[name]),
            "git_commit": git_sha(), "run_utc": run_stamp,
            "epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz,
            "device": args.device, "seed": args.seed,
            "deterministic": not is_transformer,
            "dataset": check,
            "locked_test_evaluated": False,
        }
        (run_dir / "experiment_manifest.json").write_text(
            json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Development run complete: {run_dir}")
    print("DO NOT rank models by historical metrics or evaluate the locked test "
          "until all models, thresholds, and reporting rules are frozen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
