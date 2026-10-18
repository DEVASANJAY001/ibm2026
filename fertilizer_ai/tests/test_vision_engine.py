"""
tests/test_vision_engine.py
───────────────────────────
Tests for the Hugging Face / vision deficiency engine.
"""

from PIL import Image
from vision.deficiency_engine import predict_deficiency, _map_hf_label_to_advice


def test_hf_advice_mapping():
    # Healthy
    adv = _map_hf_label_to_advice("Healthy Corn (Maize) Plant", 0.95)
    assert adv["prediction"] == "Healthy"
    assert adv["severity"] == "none"
    assert adv["nutrient"] is None

    # Scorch / Potassium
    adv = _map_hf_label_to_advice("Strawberry with Leaf Scorch", 0.88)
    assert adv["prediction"] == "Potassium_Deficiency"
    assert adv["nutrient"] == "K2O"
    assert "Potassium" in adv["symptom"]

    # Fungal Disease
    adv = _map_hf_label_to_advice("Tomato with Early Blight", 0.92)
    assert adv["prediction"] == "Disease_Stress"
    assert "Fungicide" in adv["action"] or "fungicide" in adv["action"]


def test_predict_deficiency_with_pil_image():
    # Dummy synthetic leaf
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    result = predict_deficiency(img, crop="Maize")

    assert result["status"] == "ok"
    assert "prediction" in result
    assert "diagnosis" in result
    assert "confidence" in result
    assert "action" in result
    assert len(result["top_k"]) > 0
    assert "HuggingFace" in result["model_version"] or "YOLOv8" in result["model_version"]
