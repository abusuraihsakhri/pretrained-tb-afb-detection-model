#!/usr/bin/env python3
"""Prepare available datasets and run development training."""
import gc
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = BASE_DIR / "02_CODE" / "scripts"
PROCESSED_TILES_DIR = BASE_DIR / "01_DATA" / "processed_tiles"


def log_step(message):
    print(f"\n{message}")


def clear_cache_and_memory():
    """Remove cached YOLO dataset indexes and collect Python garbage."""
    for cache_file in PROCESSED_TILES_DIR.rglob("*.cache"):
        try:
            cache_file.unlink()
            print(f"Removed cache: {cache_file.name}")
        except OSError as error:
            print(f"Could not remove {cache_file.name}: {error}", file=sys.stderr)
    gc.collect()


def run_with_retry(cmd_list, max_retries=3, cwd=str(BASE_DIR)):
    """Retry a failed command with a longer delay each time."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[*] Executing: {' '.join(cmd_list)}")
            subprocess.run(cmd_list, cwd=cwd, check=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Command failed with exit code: {e.returncode}")
            if attempt < max_retries:
                wait_time = 5 * attempt
                print(f"[*] Retrying in {wait_time} seconds (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"[FATAL] Max retries reached for: {' '.join(cmd_list)}. Halting pipeline.")
                return False
        except Exception as e:
            print(f"[FATAL] Unexpected crash: {e}")
            return False


def check_disk_space(min_gb=15.0):
    """Check free disk space before preparing the dataset."""
    free = shutil.disk_usage(BASE_DIR).free
    free_gb = free / (2**30)
    if free_gb < min_gb:
        print(f"[FATAL] Disk space critically low ({free_gb:.2f} GB). Need at least {min_gb} GB.")
        return False
    return True


def auto_pipeline():
    log_step("Checking free disk space")
    if not check_disk_space(min_gb=10.0):
        sys.exit(1)

    clear_cache_and_memory()

    log_step("Checking whether to import Roboflow datasets")

    train_images_dir = PROCESSED_TILES_DIR / "train" / "images"
    existing_images = len(list(train_images_dir.glob("*.jpg"))) if train_images_dir.exists() else 0

    env_file = BASE_DIR / ".env"
    if existing_images > 4000:
        print(f"[*] Detected {existing_images} research images. Skipping redundant download.")
    elif env_file.exists():
        print("[*] Found .env. Importing public datasets...")
        run_with_retry([sys.executable, str(SCRIPTS_DIR / "05_ingest_public_data.py")], max_retries=2)
    else:
        print("[*] No .env file. Skipping Roboflow public datasets.")

    clear_cache_and_memory()

    log_step("Starting development training")

    success = run_with_retry([
        sys.executable, str(SCRIPTS_DIR / "02_train.py"),
        "--data", "data.yaml",
        "--epochs", "150"
    ], max_retries=3)

    if success:
        log_step("Development training finished. Evaluation on the locked test set is a separate step.")
    else:
        print("\n[FATAL] The training command failed.")
        sys.exit(1)


if __name__ == "__main__":
    auto_pipeline()
