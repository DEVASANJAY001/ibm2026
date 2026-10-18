"""
vision/dataset_prep.py
───────────────────────
Step 1 of the Vision Engine pipeline.

Downloads PlantVillage from Hugging Face, filters to the 8 target crops,
maps disease labels → deficiency categories (N, P, K, Healthy, Disease),
and organises images into YOLOv8 classification folder structure:

    vision/data/npk_dataset/
        train/
            Nitrogen_Deficiency/
            Phosphorus_Deficiency/
            Potassium_Deficiency/
            Disease_Stress/
            Healthy/
        val/
            ...
        test/
            ...

Usage:
    python -m vision.dataset_prep
    python -m vision.dataset_prep --limit 500   # quick smoke-test with 500 images

The PlantVillage dataset (mohanty/PlantVillage) has 54,306 colour images
across 38 plant-disease classes. We keep only classes relevant to our 8 crops
and remap them to NPK deficiency categories based on known symptom patterns.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# ─── Label mapping: PlantVillage class name → our deficiency category ─────────
# Source: ICAR / literature on NPK deficiency visual symptoms
# N deficiency  → generalised yellowing (chlorosis of older leaves)
# P deficiency  → purple/dark-green discolouration, stunted
# K deficiency  → marginal leaf scorch / necrosis on older leaves
# Disease_Stress → fungal/bacterial lesions (separate from nutrient)
# Healthy        → no visible stress

LABEL_MAP: Dict[str, str] = {
    # Rice
    "Rice___Leaf_scald": "Disease_Stress",
    "Rice___Brown_spot": "Disease_Stress",
    "Rice___Neck_rot": "Disease_Stress",
    "Rice___healthy": "Healthy",
    # Tomato
    "Tomato___Bacterial_spot": "Disease_Stress",
    "Tomato___Early_blight": "Disease_Stress",
    "Tomato___Late_blight": "Disease_Stress",
    "Tomato___Leaf_Mold": "Disease_Stress",
    "Tomato___Septoria_leaf_spot": "Disease_Stress",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Disease_Stress",
    "Tomato___Target_Spot": "Disease_Stress",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Disease_Stress",
    "Tomato___Tomato_mosaic_virus": "Disease_Stress",
    "Tomato___healthy": "Healthy",
    # Potato
    "Potato___Early_blight": "Disease_Stress",
    "Potato___Late_blight": "Disease_Stress",
    "Potato___healthy": "Healthy",
    # Maize (Corn)
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Disease_Stress",
    "Corn_(maize)___Common_rust_": "Disease_Stress",
    "Corn_(maize)___Northern_Leaf_Blight": "Disease_Stress",
    "Corn_(maize)___healthy": "Healthy",
    # Wheat
    "Wheat___Leaf_rust": "Disease_Stress",
    "Wheat___Septoria": "Disease_Stress",
    "Wheat___Yellow_rust_(Stripe_rust)": "Disease_Stress",
    "Wheat___healthy": "Healthy",
    # Cotton (PlantVillage doesn't have cotton — supplementary dataset needed)
    # Groundnut / Sugarcane — not in PlantVillage; supplement with Roboflow
}

# NPK deficiency classes added via supplementary datasets (Roboflow / Mendeley)
# These are not in PlantVillage but are needed for our system.
# They are populated in vision/data/supplementary/ manually or via roboflow.
NPK_CLASSES = [
    "Nitrogen_Deficiency",    # Yellow-green older leaves
    "Phosphorus_Deficiency",  # Purple/dark tint, dull green
    "Potassium_Deficiency",   # Marginal scorch, brown edges
]

ALL_CLASSES = NPK_CLASSES + ["Disease_Stress", "Healthy"]

# Train / val / test split ratios
SPLITS = {"train": 0.70, "val": 0.20, "test": 0.10}


def download_plantvillage(out_dir: Path, limit: int = 0) -> Path:
    """
    Download mohanty/PlantVillage from Hugging Face datasets.
    Returns path to the raw images directory.
    """
    print("📥 Downloading PlantVillage from Hugging Face...")
    print("   Dataset: mohanty/PlantVillage (colour split)")
    print("   ~54,000 images, ~1.5 GB\n")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("Run:  pip install datasets pillow")
        sys.exit(1)

    try:
        from PIL import Image as PILImage
    except ImportError:
        print("ERROR: 'pillow' package not installed.")
        print("Run:  pip install pillow")
        sys.exit(1)

    ds = load_dataset("mohanty/PlantVillage", "color", trust_remote_code=True)
    raw_dir = out_dir / "raw_plantvillage"
    raw_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for split_name, split_data in ds.items():
        split_dir = raw_dir / split_name
        split_dir.mkdir(exist_ok=True)
        for i, item in enumerate(split_data):
            if limit and total >= limit:
                break
            label = item.get("label", "unknown")
            # Get label name from features
            label_name = split_data.features["label"].int2str(label)
            label_dir = split_dir / label_name
            label_dir.mkdir(exist_ok=True)
            img = item["image"]
            img.save(label_dir / f"{i:06d}.jpg")
            total += 1
            if total % 500 == 0:
                print(f"  Saved {total} images...")

    print(f"  Done: {total} images saved to {raw_dir}\n")
    return raw_dir


def reorganise_for_yolo(
    raw_dir: Path,
    out_dir: Path,
    label_map: Dict[str, str],
    limit: int = 0,
) -> Path:
    """
    Reorganise PlantVillage images into YOLOv8 classification structure.
    Skips classes not in label_map.
    """
    import random

    dataset_dir = out_dir / "npk_dataset"

    # Create output split/class dirs
    for split in SPLITS:
        for cls in ALL_CLASSES:
            (dataset_dir / split / cls).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    moved = 0
    skipped = 0

    for split_dir in raw_dir.iterdir():
        if not split_dir.is_dir():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            target_class = label_map.get(class_name)
            if target_class is None:
                skipped += 1
                continue

            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            random.shuffle(images)
            if limit:
                images = images[:max(1, limit // len(label_map))]

            n = len(images)
            n_train = int(n * SPLITS["train"])
            n_val   = int(n * SPLITS["val"])

            for idx, img_path in enumerate(images):
                if idx < n_train:
                    dest_split = "train"
                elif idx < n_train + n_val:
                    dest_split = "val"
                else:
                    dest_split = "test"

                dest = dataset_dir / dest_split / target_class / img_path.name
                shutil.copy2(img_path, dest)
                moved += 1

    print(f"  Organised: {moved} images across {len(ALL_CLASSES)} classes")
    print(f"  Skipped classes not in label_map: {skipped}")
    print(f"  Output: {dataset_dir}\n")
    return dataset_dir


def print_dataset_summary(dataset_dir: Path):
    """Print class distribution per split."""
    print("📊 Dataset Summary:")
    print(f"{'Class':<30} {'train':>7} {'val':>7} {'test':>7} {'total':>7}")
    print("-" * 60)
    for cls in ALL_CLASSES:
        counts = []
        for split in ["train", "val", "test"]:
            d = dataset_dir / split / cls
            n = len(list(d.glob("*.*"))) if d.exists() else 0
            counts.append(n)
        total = sum(counts)
        print(f"  {cls:<28} {counts[0]:>7} {counts[1]:>7} {counts[2]:>7} {total:>7}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NPK deficiency dataset")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max images to process (0 = all). Use 500 for quick test.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if raw_plantvillage already exists")
    args = parser.parse_args()

    _HERE = Path(__file__).parent
    data_dir = _HERE / "data"
    data_dir.mkdir(exist_ok=True)

    raw_dir = data_dir / "raw_plantvillage"

    if args.skip_download and raw_dir.exists():
        print(f"⏭  Skipping download — using existing {raw_dir}")
    else:
        raw_dir = download_plantvillage(data_dir, limit=args.limit)

    dataset_dir = reorganise_for_yolo(raw_dir, data_dir, LABEL_MAP, limit=args.limit)
    print_dataset_summary(dataset_dir)

    print("✅ Dataset ready for training.")
    print("   Next step:  python -m vision.train")
