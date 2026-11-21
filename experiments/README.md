# AFB: controlled CNN versus Transformer comparison

**Status:** Experiment scaffold, not a training result or clinical validation.

The historical `03_MODELS/best.pt` and the original pooled data are on
**validation hold** (see [MODEL_CARD](../MODEL_CARD.md) and
[DATA_CARD](../DATA_CARD.md)). The Git repository deliberately excludes the
images. **Do not run training on the historical aggregate or use its metrics
as a model-selection baseline.**

## Research question

Among models trained on the same carefully curated microscopy data,
does a higher-resolution CNN detection head or a hybrid Transformer improve
AFB candidate detection and false-positive burden?

| Experiment ID | Architecture | Initialization |
| --- | --- | --- |
| `yolov8n` | CNN reference | Local verified COCO-pretrained `yolov8n.pt` |
| `yolov8s` | Larger CNN | Local verified COCO-pretrained `yolov8s.pt` |
| `yolov8s_p2` | High-resolution (stride 4) CNN | `yolov8s-p2.yaml` + compatible pretrained `yolov8s.pt` weights |
| `rtdetr_l` | Hybrid CNN–Transformer detector | Local verified `rtdetr-l.pt` |

The P2 head uses the Ultralytics YOLOv8 P2 architecture and should report
strides **4, 8, 16, 32**. The pretrained standard YOLO head does not
fully transfer to P2: confirm the reported transferred layer count.
RT-DETR-L is **much larger** and may require substantially more GPU
memory and a smaller batch size. Its decoder has a finite object-query
budget; assess high-bacillary-load tiles separately.

## Required steps (local machine with research data)

1. Reconstruct upstream class meanings, licensing and provenance. Quarantine
   `afb-detect` and investigate sources with zero converted labels.
2. Generate a source inventory using the existing project command:

   ```bash
   python 02_CODE/scripts/prepare_curated_dataset.py inventory \
     --dataset-root 01_DATA/processed_tiles \
     --output 06_LOGS/reviewed_inventory.csv
   ```

3. **Manually review** each row. Verify
   `source_id`, `source_version`, `license_id`,
   `provenance_status`, `biological_group_id`, `target_split`,
   `include`, and `exclusion_reason`.
   Group all tiles from a patient/specimen/slide in one split, and reserve
   an entirely independent source/site as test. Do not fabricate missing
   group metadata.
4. Materialize the reviewed dataset using the repository's existing builder.
   Substitute a genuinely independent source for `YOUR_HELD_OUT_SOURCE`:

   ```bash
   python 02_CODE/scripts/prepare_curated_dataset.py build \
     --dataset-root 01_DATA/processed_tiles \
     --manifest 06_LOGS/reviewed_inventory.csv \
     --output-root 01_DATA/curated_afb \
     --test-source YOUR_HELD_OUT_SOURCE
   ```

5. Create `02_CODE/curated_afb.yaml` with:

   ```yaml
   path: ../01_DATA/curated_afb
   train: train/images
   val: val/images
   test: test/images
   nc: 1
   names:
     0: AFB
   ```

6. Acquire official Ultralytics pretrained weights on a trusted machine:
   `yolov8n.pt`, `yolov8s.pt`, `rtdetr-l.pt`. Pin their SHA-256 hashes
   before the experiment. Put them in a local `--weights-dir`; the runner
   will **not** silently download weights.
7. Preflight the data and provenance gate (never trains):

   ```bash
   python experiments/compare_architectures.py \
     --data 02_CODE/curated_afb.yaml \
     --source-root 01_DATA/processed_tiles \
     --reviewed-manifest 06_LOGS/reviewed_inventory.csv \
     --test-source YOUR_HELD_OUT_SOURCE
   ```

8. If and only if the gate passes, train selected candidates:

   ```bash
   python experiments/compare_architectures.py \
     --data 02_CODE/curated_afb.yaml \
     --source-root 01_DATA/processed_tiles \
     --reviewed-manifest 06_LOGS/reviewed_inventory.csv \
     --test-source YOUR_HELD_OUT_SOURCE \
     --weights-dir /path/to/verified/weights \
     --model yolov8n --model yolov8s --model yolov8s_p2 \
     --epochs 100 --imgsz 640 --batch 2 --device 0 --train
   ```

   Run `--model rtdetr_l` separately with a GPU and batch size that fit its
   memory use. Never use the already-trained, leakage-affected
   `03_MODELS/best.pt` as a starting point.

## Evaluation and study design

- **Development only:** the experiment runner calls Ultralytics `train()`
  using **train + validation**, never the test split. Its output is explicitly
  labeled as development data. Train with predeclared architecture and seed.
- **Operating points:** choose a confidence threshold for each model using
  development validation data. Fix the primary false-positive budget before
  looking at test performance.
- **Locked test:** after all candidates and thresholds are frozen, evaluate
  each final checkpoint *once* on the held-out source using the existing
  `02_CODE/scripts/03_evaluate_locked.py` workflow. Save model hashes,
  predeclared endpoints, calibration, and a test manifest.
- **Primary scientific outcome:** candidate-level recall at a prespecified
  false-positive rate per image or correctly calibrated high-power field.
- **Secondary:** AP50, AP50:95, size-stratified recall, processing time and
  memory use; report clustered CIs at a specimen/slide level. Do not equate
  object-level precision with specimen-level diagnostic specificity.
- **Stratify:** low-bacillary-load fields, stain artifacts, scanner or
  microscope, source/site and magnification. Field-of-view calibration is
  required before reporting counts per high-power field.
- **Smear-level performance:** requires a separate, prespecified field
  sampling protocol and independently verified specimen reference standard.
  AFB candidate counts are not direct microbiological tuberculosis diagnoses.

**Comparison caveat:** a strict 640-pixel, 100-epoch comparison is a
controlled starting experiment, not a guarantee of equal optimization.
Conduct sensitivity analyses for input resolution, training stability,
independent seeds and model-specific hyperparameters before asserting
architecture superiority.

For smoke testing the **code-only gates**:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q tests/test_architecture_comparison.py
python experiments/compare_architectures.py --list-models
```

The tests create tiny synthetic PNG files for dataset-validation mechanics;
they provide **no evidence of medical detection performance**.

## Sources

- [Ultralytics YOLOv8 P2 config](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8-p2.yaml)
- [RT-DETR model/train documentation](https://docs.ultralytics.com/models/rtdetr/)
- [YOLOv8 model guide](https://docs.ultralytics.com/models/yolov8/)
