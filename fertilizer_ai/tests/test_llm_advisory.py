"""
tests/test_llm_advisory.py
──────────────────────────
Unit tests for the grounded LLM Advisory Layer with Qwen 2.5 and Ollama.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.advisory_engine import (
    check_llm_status,
    generate_advisory,
    _build_system_prompt,
    _rule_based_advisory,
)


def sample_recommendation():
    return {
        "crop": "Rice",
        "growth_stage": "Vegetative",
        "area_ha": 1.5,
        "nutrients_kg_ha": {"N": 105.0, "P2O5": 45.0, "K2O": 75.0},
        "npk_ratio": "2.3:1.0:1.7",
        "confidence": "high",
        "products": {
            "dap": {"bags": 2, "total_kg": 97.8, "cost_inr": 2700},
            "urea": {"bags": 4, "total_kg": 180.2, "cost_inr": 1072},
            "mop": {"bags": 3, "total_kg": 125.0, "cost_inr": 5100},
        },
        "total_cost_inr": 8872,
        "schedule": [
            {"stage": "Basal", "nutrients": {"dap": {"kg_field": 97.8}}},
            {"stage": "Tillering", "nutrients": {"urea": {"kg_field": 90.0}}},
        ],
        "warnings": ["Top-dress N after heavy rain subsides."],
        "leaf_diagnosis": {
            "scanned": True,
            "prediction": "Potassium_Deficiency",
            "confidence_pct": "89%",
            "status": "Deficiency Detected (Potassium Scorch)",
            "impact": "Potassium (MOP) dose boosted by +25% to treat foliar scorch.",
        },
        "vision_adjustment": {
            "applied": True,
            "type": "Potassium_Deficiency_Recovery",
            "detail": "Potassium (MOP) dose boosted by +25% to treat foliar scorch.",
        },
    }


def test_system_prompt_grounding():
    prompt_en = _build_system_prompt("en")
    assert "Tamil Nadu" in prompt_en
    assert "CRITICAL RULES" in prompt_en
    assert "NEVER invent" in prompt_en

    prompt_ta = _build_system_prompt("ta")
    assert "Tamil" in prompt_ta


def test_llm_status_check():
    status = check_llm_status()
    assert "status" in status
    assert "provider" in status
    assert "default_model" in status
    assert "available_models" in status
    assert "qwen_ready" in status


def test_rule_based_advisory_contains_leaf_and_bags():
    data = sample_recommendation()
    text = _rule_based_advisory(data)
    assert "RICE" in text
    assert "LEAF DIAGNOSIS" in text
    assert "Potassium_Deficiency" in text
    assert "DAP" in text
    assert "UREA" in text
    assert "MOP" in text
    assert "8,872" in text or "8872" in text


def test_generate_advisory_fallback():
    data = sample_recommendation()
    adv = generate_advisory(data, provider="rule_based", lang="en")
    assert adv["provider"] == "rule_based"
    assert adv["language"] == "en"
    assert len(adv["text"]) > 50


def test_generate_advisory_tamil_flag():
    data = sample_recommendation()
    adv = generate_advisory(data, provider="rule_based", lang="ta")
    assert adv["language"] == "ta"
