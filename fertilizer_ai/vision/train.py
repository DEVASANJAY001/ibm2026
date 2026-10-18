"""
vision/train.py
───────────────
Step 2: Fine-tune YOLOv8n-cls on the NPK deficiency dataset.

What this trains:
  - Model:   YOLOv8n-cls (Nano, ~3 MB)  — fastest, runs on CPU
  - Classes: Nitrogen_Deficiency, Phosphorus_Deficiency,
             Potassium_Deficiency, Disease_Stress, Healthy
  - Input:   vision/data/npk_dataset/  (built by dataset_prep.py)
  - Output:  vision/models/npk_best.pt

Why YOLOv8n-cls (not detect)?
  PlantVillage gives whole-leaf images with one label per image.
  Classification mode is correct here and trains 5× faster than detection.
  Upgrade to YOLOv8s-detect when you have bounding-box annotations.

Usage:
    python -m vision.train
    python -m vision.train --epochs 50 --device cpu
    python -m vision.train --resume   # resume from last checkpoint
"""

import argparse
import shutil
from pathlib import Path

_HERE = Path(__file__).parent
DATASET_DIR = _HERE / "data" / "npk_dataset"
MODELS_DIR  = _HERE / "models"
MODELS_DIR.mkdir(exist_ok=True)


def check_dataset():
    if not DATASET_DIR.exists():
        print("❌ Dataset not found. Run first:")
        print("   python -m vision.dataset_prep")
        return False

    classes = [d.name for d in (DATASET_DIR / "train").iterdir() if d.is_dir()]
    if not classes:
        print("❌ No class folders found in vision/data/npk_dataset/train/")
        return False

    print(f"✅ Dataset found: {len(classes)} classes")
    for cls in sorted(classes):
        n = len(list((DATASET_DIR / "train" / cls).glob("*.*")))
        print(f"   {cls}: {n} train images")
    return True


def train(epochs: int = 50, imgsz: int = 224, batch: int = 32,
          device: str = "0", resume: bool = False):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run:  pip install ultralytics")
        return

    if not check_dataset():
        return

    # Load pretrained nano classification model
    model_name = "yolov8n-cls.pt"
    print(f"\n🚀 Training YOLOv8n-cls on NPK deficiency dataset")
    print(f"   epochs={epochs}  imgsz={imgsz}  batch={batch}  device={device}\n")

    model = YOLO(model_name)

    results = model.train(
        data=str(DATASET_DIR),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(_HERE / "runs"),
        name="npk_cls",
        exist_ok=resume,
        patience=10,           # early stopping
        optimizer="AdamW",
        lr0=1e-3,
        weight_decay=5e-4,
        augment=True,
        flipud=0.3,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        verbose=True,
    )

    # Copy best weights to models/
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = MODELS_DIR / "npk_best.pt"
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"\n✅ Best model saved → {best_dst}")
    else:
        print("⚠  best.pt not found in run directory")

    # Print metrics
    print(f"\n📊 Training Results:")
    print(f"   Top-1 Accuracy: {results.results_dict.get('metrics/accuracy_top1', 'N/A')}")
    print(f"   Top-5 Accuracy: {results.results_dict.get('metrics/accuracy_top5', 'N/A')}")
    print(f"   Run saved at:   {results.save_dir}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n-cls for NPK deficiency")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu",
                        help="'cpu', '0' (GPU 0), 'mps' (Apple Silicon)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        resume=args.resume,
    )
