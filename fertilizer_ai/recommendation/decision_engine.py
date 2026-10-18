"""
recommendation/decision_engine.py
───────────────────────────────────
Orchestrates all engines and produces the final structured JSON response.

Pipeline:
  1. Nutrient Engine  → N, P2O5, K2O kg/ha + schedule
  2. Health Proxy     → crop health flags
  3. Weather Engine   → rain/temp advisories
  4. Rule Checks      → yield ambition, weather delay, confidence floor
  5. ML Cross-check   → disagree warning (never overrides formula)
  6. Fertilizer Calculator → products, bags, cost
  7. LLM Advisory     → farmer-readable plain-language output (async)

Every response carries: method, confidence, trace.
"""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any

_HERE = Path(__file__).parent.parent
_RULES_PATH = _HERE / "config" / "rules.yaml"

with open(_RULES_PATH, "r", encoding="utf-8") as _f:
    _RULES = yaml.safe_load(_f)

# ─── Confidence ordering ──────────────────────────────────────────────────────
_CONF_RANK = {"high": 3, "medium": 2, "low": 1, "none": 0}


def _lower_confidence(current: str, candidate: str) -> str:
    return current if _CONF_RANK[current] <= _CONF_RANK[candidate] else candidate


# ─── ML cross-check (lazy import) ────────────────────────────────────────────
_ML_MODEL = None


def _load_ml():
    global _ML_MODEL
    if _ML_MODEL is not None:
        return _ML_MODEL
    try:
        import joblib
        from ml.bundle import FertilizerModelBundle  # noqa
        m = joblib.load(_HERE / "ml" / "fert_classifier.joblib")
        _ML_MODEL = m
        return m
    except Exception:
        return None


_DOMINANT_LABEL = {
    "N":    "Nitrogen_Fertilizer",
    "P2O5": "Phosphorus_Fertilizer",
    "K2O":  "Potassium_Fertilizer",
}


def _ml_crosscheck(row_dict: dict, dominant_nutrient: str) -> dict:
    """
    Run the ML classifier and compare its prediction to the formula dominant nutrient.
    Returns {agree, ml_label, note}.
    """
    model = _load_ml()
    if model is None:
        return {"available": False, "note": "ML model not loaded (train first)."}

    try:
        import pandas as pd
        from ml.features import FEATURES, CAT  # noqa

        X = pd.DataFrame([row_dict])[FEATURES]
        pred_idx = model.predict(X)[0]
        label = model.classes_[pred_idx] if hasattr(model, "classes_") else str(pred_idx)
        agree = label == _DOMINANT_LABEL.get(dominant_nutrient, "")
        return {
            "available": True,
            "ml_label": label,
            "formula_dominant": _DOMINANT_LABEL.get(dominant_nutrient, "unknown"),
            "agree": agree,
            "note": "" if agree else _RULES["ml"]["disagree_note"],
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


# ─── Main entry point ─────────────────────────────────────────────────────────

def recommend(
    # Required
    crop: str,
    growth_stage: str,
    soil_type: str,
    irrigation_type: str,
    area_ha: float,
    target_yield: float,
    soil_n: Optional[float],
    soil_p: Optional[float],
    soil_k: Optional[float],
    soil_ph: float,
    # Optional – weather
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    # Optional – context
    previous_yield: Optional[float] = None,
    organic_matter: Optional[float] = None,
    ec: Optional[float] = None,
    crop_age_days: Optional[float] = None,
    air_temp: Optional[float] = None,
    humidity: Optional[float] = None,
    rainfall: Optional[float] = None,
    # Optional – health proxy
    chlorophyll: Optional[float] = None,
    stress: Optional[float] = None,
    pest: Optional[float] = None,
    disease: Optional[float] = None,
    weed: Optional[float] = None,
    # Optional – product preference
    p_source: str = "dap",
    # Computer Vision Leaf Screening
    vision_label: Optional[str] = None,
    vision_confidence: Optional[float] = None,
) -> Dict[str, Any]:

    warnings: list = []
    trace: list = []
    confidence = "high"

    # ── 1. Nutrient Engine ────────────────────────────────────────────────
    from engines.nutrient_engine import compute_dose  # noqa

    nutrient_result = compute_dose(
        crop=crop,
        target_yield=target_yield,
        soil={"N": soil_n, "P": soil_p, "K": soil_k},
        area_ha=area_ha,
    )
    trace += nutrient_result.get("trace", [])

    if nutrient_result["status"] != "ok":
        return {
            "status": "error",
            "stage": "nutrient_engine",
            "error": nutrient_result,
            "trace": trace,
        }

    confidence = _lower_confidence(confidence, nutrient_result["confidence"])
    n_kg = nutrient_result["N_kg_ha"]
    p_kg = nutrient_result["P2O5_kg_ha"]
    k_kg = nutrient_result["K2O_kg_ha"]

    # ── 2. Yield ambition check ───────────────────────────────────────────
    ambition_ratio = _RULES["yield"]["ambition_ratio"]
    if previous_yield and previous_yield > 0 and target_yield > ambition_ratio * previous_yield:
        warnings.append(
            f"Target yield ({target_yield} t/ha) is >{ambition_ratio}× previous yield "
            f"({previous_yield} t/ha). Consider a more conservative target."
        )
        trace.append(f"Yield ambition warning: {target_yield} > {ambition_ratio}×{previous_yield}")
        confidence = _lower_confidence(confidence, "low")

    # ── 3. Health Proxy ───────────────────────────────────────────────────
    from engines.health_engine import health_flags  # noqa

    health_result = health_flags(
        crop=crop,
        stage=growth_stage,
        chlorophyll=chlorophyll,
        stress=stress,
        pest=pest,
        disease=disease,
        weed=weed,
    )
    trace += health_result.get("trace", [])
    # Only factor in health confidence if it was actually assessed
    if health_result["status"] == "assessed":
        confidence = _lower_confidence(confidence, health_result["confidence"])

    # Health flags NEVER raise dose – only add advice
    for flag in health_result.get("flags", []):
        if flag.get("severity") in ("high", "moderate") and flag["issue"] != "none_detected":
            warnings.append(
                f"Health flag [{flag['issue']} – {flag['severity']}]: {flag.get('next_step', '')}"
            )

    # ── 4. Weather Engine ─────────────────────────────────────────────────
    weather_result: Dict = {}
    if lat is not None and lon is not None:
        from engines.weather_engine import get_weather  # noqa
        weather_result = get_weather(lat, lon)
        trace += weather_result.get("trace", [])

        if weather_result.get("source") == "unavailable":
            warnings.append("Weather data unavailable. Check local forecast before applying fertilizer.")
            confidence = _lower_confidence(confidence, "medium")
        else:
            for adv in weather_result.get("advisories", []):
                warnings.append(adv["message"])
    else:
        weather_result = {"source": "not_requested"}
    # ── 4b. Computer Vision Leaf Screening Integration ──────────────────
    conf_pct = round(vision_confidence * 100, 1) if vision_confidence else None
    conf_str = f"{conf_pct}%" if conf_pct is not None else "N/A"

    vision_adjustment = {
        "applied": False,
        "type": "Standard_Baseline",
        "diagnosis": vision_label or "Not scanned",
        "confidence": conf_pct,
        "detail": "Standard baseline calculation without visual leaf anomaly."
    }
    leaf_status = "Healthy / Normal"

    if vision_label:
        v_lbl = vision_label.lower()
        if "potassium" in v_lbl or "scorch" in v_lbl:
            k_old = k_kg
            k_kg = round(k_kg * 1.25, 1)  # +25% Potassium boost to treat foliar scorch
            leaf_status = "Deficiency Detected (Potassium Scorch)"
            vision_adjustment = {
                "applied": True,
                "type": "Potassium_Deficiency_Recovery",
                "diagnosis": vision_label,
                "confidence": conf_pct,
                "detail": f"Leaf scorch detected. Potassium (MOP) dose boosted by +25% ({k_old} -> {k_kg} kg/ha) to overcome visual foliar deficiency."
            }
            warnings.append(vision_adjustment["detail"])
            trace.append(f"Vision AI adjusted K2O from {k_old} to {k_kg} kg/ha due to leaf scorch.")
        elif "nitrogen" in v_lbl:
            n_old = n_kg
            n_kg = round(n_kg * 1.15, 1)  # +15% Nitrogen boost for chlorosis
            leaf_status = "Deficiency Detected (Nitrogen Chlorosis)"
            vision_adjustment = {
                "applied": True,
                "type": "Nitrogen_Deficiency_Recovery",
                "diagnosis": vision_label,
                "confidence": conf_pct,
                "detail": f"Nitrogen chlorosis detected on older leaves. N dose increased by +15% ({n_old} -> {n_kg} kg/ha)."
            }
            warnings.append(vision_adjustment["detail"])
            trace.append(f"Vision AI adjusted N from {n_old} to {n_kg} kg/ha due to chlorosis.")
        elif any(dis in v_lbl for dis in ["blight", "rust", "spot", "mildew", "rot", "disease"]):
            n_old = n_kg
            n_kg = round(n_kg * 0.85, 1)  # Reduce Nitrogen by -15% to suppress fungal spread
            leaf_status = "Pathogen / Disease Stress Detected"
            vision_adjustment = {
                "applied": True,
                "type": "Disease_Protection_N_Reduction",
                "diagnosis": vision_label,
                "confidence": conf_pct,
                "detail": f"Fungal/bacterial disease detected ({vision_label}). Nitrogen restricted by -15% ({n_old} -> {n_kg} kg/ha) to prevent pathogen proliferation. Apply crop fungicide."
            }
            warnings.append(vision_adjustment["detail"])
            trace.append(f"Vision AI restricted N from {n_old} to {n_kg} kg/ha to avoid disease flare-up.")
        elif "healthy" in v_lbl:
            leaf_status = "Verified Healthy Foliage"
            vision_adjustment = {
                "applied": True,
                "type": "Healthy_Confirmed",
                "diagnosis": vision_label,
                "confidence": conf_pct,
                "detail": "Leaf scan verified healthy foliage. Standard STCR balanced dose maintained."
            }
            trace.append("Vision AI verified healthy foliage. No dose distortion needed.")

    # ── 5. ML Cross-check ─────────────────────────────────────────────────
    dominant = max({"N": n_kg, "P2O5": p_kg, "K2O": k_kg}, key=lambda x: {"N": n_kg, "P2O5": p_kg, "K2O": k_kg}[x])

    ml_input = {
        "Crop_Type": crop,
        "Soil_Type": soil_type,
        "Growth_Stage": growth_stage,
        "Irrigation_Type": irrigation_type,
        "Nitrogen_mg_kg": soil_n or 0,
        "Phosphorus_mg_kg": soil_p or 0,
        "Potassium_mg_kg": soil_k or 0,
        "Soil_pH": soil_ph,
        "Soil_Organic_Matter_Percent": organic_matter,
        "Electrical_Conductivity_dS_m": ec,
        "Target_Yield_ton_per_Hectare": target_yield,
        "Previous_Yield_ton_per_Hectare": previous_yield,
        "Crop_Age_Days": crop_age_days,
        "Air_Temperature_C": air_temp,
        "Relative_Humidity_Percent": humidity,
        "Rainfall_mm": rainfall,
    }

    ml_check = _ml_crosscheck(ml_input, dominant)
    if ml_check.get("available") and not ml_check.get("agree"):
        warnings.append(ml_check["note"])
        confidence = _lower_confidence(confidence, "medium")
        trace.append(
            f"ML disagrees: formula={ml_check['formula_dominant']}  ML={ml_check['ml_label']}"
        )

    # ── 6. Fertilizer Calculator ──────────────────────────────────────────
    from recommendation.fertilizer_calculator import calc_products  # noqa

    fert_result = calc_products(
        n_kg_ha=n_kg,
        p2o5_kg_ha=p_kg,
        k2o_kg_ha=k_kg,
        area_ha=area_ha,
        p_source=p_source,
    )
    trace += fert_result.get("trace", [])
    warnings += fert_result.get("notes", [])

    # ── 7. Final confidence (lowest across all engines) ───────────────────
    if health_result["status"] == "not_assessed":
        trace.append("Health proxy not assessed (no indices provided).")

    # ── 8. Assemble response ──────────────────────────────────────────────
    return {
        "status": "ok",
        "crop": crop,
        "growth_stage": growth_stage,
        "area_ha": area_ha,
        "target_yield_t_ha": target_yield,

        # Core nutrient doses
        "nutrients_kg_ha": {
            "N": n_kg,
            "P2O5": p_kg,
            "K2O": k_kg,
        },
        "nutrients_field_kg": {
            "N": round(n_kg * area_ha, 1),
            "P2O5": round(p_kg * area_ha, 1),
            "K2O": round(k_kg * area_ha, 1),
        },
        "npk_ratio": f"{round(n_kg/min(n_kg, p_kg, k_kg), 1) if min(n_kg, p_kg, k_kg)>0 else 1.0}:{round(p_kg/min(n_kg, p_kg, k_kg), 1) if min(n_kg, p_kg, k_kg)>0 else 1.0}:{round(k_kg/min(n_kg, p_kg, k_kg), 1) if min(n_kg, p_kg, k_kg)>0 else 1.0}",

        # Method metadata
        "method": nutrient_result["method"],
        "confidence": confidence,
        "verified": nutrient_result.get("verified", False),
        "source": nutrient_result.get("source", ""),

        # Products
        "products": fert_result["products"],
        "total_cost_inr": fert_result["total_cost_inr"],
        "total_cost_min_inr": fert_result.get("total_cost_min_inr", fert_result["total_cost_inr"]),
        "total_cost_max_inr": fert_result.get("total_cost_max_inr", fert_result["total_cost_inr"]),
        "approx_total_cost_range_inr": fert_result.get("approx_total_cost_range_inr", f"₹{fert_result['total_cost_inr']:,}"),

        # Schedule
        "schedule": nutrient_result["schedule"],

        # Computer Vision Leaf Screening
        "leaf_diagnosis": {
            "scanned": bool(vision_label),
            "prediction": vision_label or "No leaf scan provided",
            "confidence": vision_confidence,
            "confidence_pct": conf_str,
            "status": leaf_status,
            "adjustment_applied": vision_adjustment.get("applied", False),
            "adjustment_type": vision_adjustment.get("type", "Standard_Baseline"),
            "impact": vision_adjustment.get("detail", "Standard baseline calculation without visual leaf anomaly."),
        },
        "vision_adjustment": vision_adjustment,

        # Health
        "health": health_result,

        # Weather
        "weather": weather_result,

        # ML cross-check summary
        "ml_check": ml_check,

        # Warnings & trace
        "warnings": warnings,
        "trace": trace,
    }
