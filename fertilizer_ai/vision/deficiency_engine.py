"""
vision/deficiency_engine.py
────────────────────────────
Image-based crop deficiency & leaf disease detection engine.

Architecture:
  • Primary: Hugging Face Pretrained Vision Model (MobileNetV2 / PlantVillage 38 classes)
             Auto-loads instantly from cache/hub with zero manual training required.
  • Optional: Fine-tuned YOLOv8n-cls (vision/models/npk_best.pt) if present.
  • CLAHE preprocessing for consistent field lighting conditions.
  • Agronomic mapping: Translates visual leaf diagnosis into actionable NPK advice.

Classes handled:
  - Nitrogen / Chlorosis stress (Yellowing leaves, virus-induced chlorosis)
  - Potassium / Scorch stress   (Marginal leaf scorch, necroses)
  - Disease stress              (Early/late blight, rust, leaf spot, scab, mold)
  - Healthy                     (No visible stress)

Usage:
    from vision.deficiency_engine import predict_deficiency
    result = predict_deficiency("path/to/leaf.jpg", crop="Tomato")
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

_HERE = Path(__file__).parent
_YOLO_PATH = _HERE / "models" / "npk_best.pt"
_LOCAL_HF_PATH = _HERE / "models" / "mobilenet_v2_plant_disease"
_HF_MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

MIN_CONFIDENCE = 0.50

# Global cached model instances
_YOLO_MODEL = None
_HF_MODEL = None


def _clahe_preprocess(img_array: np.ndarray) -> np.ndarray:
    """Apply CLAHE for lighting normalisation in field conditions."""
    try:
        import cv2
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    except Exception:
        return img_array


def _load_model() -> Tuple[Optional[str], Optional[object]]:
    """
    Load available vision model.
    1. Returns ("yolo", model) if fine-tuned npk_best.pt exists.
    2. Returns ("hf", model) using the local or cached Hugging Face vision model.
    """
    global _YOLO_MODEL, _HF_MODEL

    # 1. Check custom YOLO
    if _YOLO_PATH.exists():
        if _YOLO_MODEL is not None:
            return ("yolo", _YOLO_MODEL)
        try:
            from ultralytics import YOLO
            _YOLO_MODEL = YOLO(str(_YOLO_PATH))
            return ("yolo", _YOLO_MODEL)
        except Exception:
            pass

    # 2. Check Hugging Face model
    if _HF_MODEL is not None:
        return ("hf", _HF_MODEL)

    try:
        from transformers import AutoModelForImageClassification
        # Prefer local weights directory if available
        model_source = str(_LOCAL_HF_PATH) if _LOCAL_HF_PATH.exists() else _HF_MODEL_NAME
        _HF_MODEL = AutoModelForImageClassification.from_pretrained(model_source)
        _HF_MODEL.eval()
        return ("hf", _HF_MODEL)
    except Exception as exc:
        return (None, None)


def _map_hf_label_to_advice(label: str, confidence: float) -> Dict[str, Union[str, Optional[str]]]:
    """
    Maps 38 PlantVillage diagnoses into agronomic fertilizer and deficiency advice.
    """
    lbl_lower = label.lower()

    if "healthy" in lbl_lower:
        return {
            "prediction": "Healthy",
            "diagnosis": label,
            "nutrient": None,
            "severity": "none",
            "symptom": f"Healthy foliage detected ({label}). No visible chlorosis or lesions.",
            "action": "Proceed with standard scheduled fertilizer application based on soil test target yield.",
        }

    if "scorch" in lbl_lower:
        severity = "severe" if confidence > 0.8 else "moderate"
        return {
            "prediction": "Potassium_Deficiency",
            "diagnosis": label,
            "nutrient": "K2O",
            "severity": severity,
            "symptom": f"Marginal leaf scorch / necrosis ({label}). High risk of Potassium deficiency or moisture stress.",
            "action": "Apply MOP (Muriate of Potash). Ensure adequate irrigation — Potassium uptake is strictly water-dependent.",
        }

    if "yellow leaf curl" in lbl_lower or "citrus greening" in lbl_lower:
        severity = "severe" if confidence > 0.8 else "moderate"
        return {
            "prediction": "Disease_Stress",
            "diagnosis": label,
            "nutrient": "N",
            "severity": severity,
            "symptom": f"Viral/vector disease causing leaf yellowing and curl ({label}).",
            "action": "Rogue infected plants immediately; withhold high-nitrogen foliar sprays; control whitefly vector with neem oil/insecticide.",
        }

    # General fungal or bacterial leaf disease (Blight, Rust, Spot, Mold, Rot)
    severity = "high" if confidence > 0.8 else "medium"
    return {
        "prediction": "Disease_Stress",
        "diagnosis": label,
        "nutrient": None,
        "severity": severity,
        "symptom": f"Fungal or bacterial disease detected: {label}.",
        "action": (
            "Avoid excessive foliar Nitrogen application which accelerates pathogen spread. "
            "Apply Potassium (K2O) to strengthen leaf cuticle thickness and disease resistance, "
            "and apply crop-appropriate fungicide/bactericide."
        ),
    }


def predict_deficiency(
    image: Union[str, Path, bytes, "PIL.Image.Image"],  # noqa: F821
    crop: Optional[str] = None,
    top_k: int = 3,
) -> Dict:
    """
    Predict crop health, disease, and nutrient deficiency advice from a leaf photo.
    """
    trace = [f"Vision engine called | crop={crop}"]

    # 1. Load model
    model_type, model = _load_model()
    if model is None:
        return {
            "status": "model_not_ready",
            "message": "Vision model not loaded. Ensure internet connection to download Hugging Face model or train YOLO.",
            "trace": trace,
        }

    # 2. Process image
    try:
        from PIL import Image as PILImage

        if isinstance(image, (str, Path)):
            pil_img = PILImage.open(image).convert("RGB")
            trace.append(f"Loaded image from path: {image}")
        elif isinstance(image, bytes):
            pil_img = PILImage.open(io.BytesIO(image)).convert("RGB")
            trace.append("Loaded image from bytes")
        else:
            pil_img = image.convert("RGB")
            trace.append("Received PIL Image object")

        # Optional CLAHE lighting adjustment
        img_arr = np.array(pil_img)
        img_arr = _clahe_preprocess(img_arr)
        pil_img = PILImage.fromarray(img_arr)

    except Exception as exc:
        return {
            "status": "invalid_image",
            "error": str(exc),
            "trace": trace + [f"Image loading failed: {exc}"],
        }

    # 3. Inference
    try:
        if model_type == "yolo":
            # YOLOv8 format
            results = model.predict(pil_img, verbose=False)[0]
            probs = results.probs
            names = results.names
            top_class_idx = int(probs.top1)
            top_confidence = float(probs.top1conf)
            top_class_name = names[top_class_idx]

            top_k_indices = probs.top5[:top_k]
            top_k_confs = probs.top5conf[:top_k].tolist()
            top_k_preds = [
                {"class": names[int(idx)], "confidence": round(float(c), 4)}
                for idx, c in zip(top_k_indices, top_k_confs)
            ]
            advice = _map_hf_label_to_advice(top_class_name, top_confidence)
            model_ver = f"YOLOv8/{_YOLO_PATH.name}"

        else:
            # Hugging Face PyTorch model
            import torch
            resized = pil_img.resize((224, 224))
            arr = np.array(resized, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            norm_arr = (arr - mean) / std
            tensor = torch.from_numpy(norm_arr.transpose(2, 0, 1)).unsqueeze(0)

            with torch.no_grad():
                logits = model(tensor).logits
                probs = torch.softmax(logits, dim=-1)[0]

            top_vals, top_indices = torch.topk(probs, min(top_k, len(probs)))
            top_confidence = float(top_vals[0].item())
            top_class_idx = int(top_indices[0].item())
            raw_label = model.config.id2label[top_class_idx]

            top_k_preds = [
                {
                    "class": model.config.id2label[int(idx.item())],
                    "confidence": round(float(v.item()), 4),
                }
                for v, idx in zip(top_vals, top_indices)
            ]
            advice = _map_hf_label_to_advice(raw_label, top_confidence)
            model_ver = f"HuggingFace/{_HF_MODEL_NAME.split('/')[-1]}"

        trace.append(f"Diagnosis: {advice['diagnosis']} (conf={top_confidence:.4f})")

    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "trace": trace + [f"Inference failed: {exc}"],
        }

    uncertain = top_confidence < MIN_CONFIDENCE
    if uncertain:
        trace.append(f"Confidence {top_confidence:.2f} < threshold {MIN_CONFIDENCE} → flagged uncertain")

    return {
        "status": "ok",
        "crop": crop,
        "prediction": advice["prediction"],
        "diagnosis": advice["diagnosis"],
        "confidence": round(top_confidence, 4),
        "confidence_pct": f"{top_confidence * 100:.1f}%",
        "severity": advice["severity"],
        "uncertain": uncertain,
        "nutrient": advice["nutrient"],
        "symptom": advice["symptom"],
        "action": advice["action"],
        "top_k": top_k_preds,
        "model_version": model_ver,
        "note": "Computer vision screening is an auxiliary diagnostic tool. Validate with soil test.",
        "trace": trace,
    }


def predict_from_upload(file_bytes: bytes, crop: Optional[str] = None) -> Dict:
    """Convenience wrapper for FastAPI file uploads."""
    return predict_deficiency(file_bytes, crop=crop)
