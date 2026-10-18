"""
engines/nutrient_engine.py
──────────────────────────
Soil / Nutrient Engine for the Precision Fertilizer Recommendation system.

Implements two methods per crop (selected by config):
  • nutrient_balance  – standard FAO/ICAR nutrient-balance approach
  • stcr             – Soil Test Crop Response: F = a*T - b*S

Every output is tagged with method, confidence, source and trace.
The engine REFUSES to compute if any config value is None / missing.
All doses are clamped to [0, max_dose_kg_ha].

Soil input units: mg/kg (ppm) – same as the dataset columns.
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, Optional

# ─── Config loading ───────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.parent
_CFG_PATH = _HERE / "config" / "crop_formulas.json"

with open(_CFG_PATH, "r", encoding="utf-8") as _f:
    _CFG: Dict = json.load(_f)

# The three nutrients we calculate (in order)
NUTRIENTS = ["N", "P2O5", "K2O"]
# Maps nutrient key → soil dict key used in soil_class_thresholds
_SOIL_KEY = {"N": "N", "P2O5": "P", "K2O": "K"}


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _find_missing(cfg: dict) -> list[str]:
    """Walk the formula config and return paths of any None values."""
    required_keys = [
        "uptake_kg_per_tonne",
        "efficiency",
        "soil_supply_fraction",
        "soil_class_thresholds",
        "max_dose_kg_ha",
    ]

    def _walk(d: dict, path: str = "") -> list[str]:
        gaps = []
        for k, v in d.items():
            full = f"{path}{k}"
            if isinstance(v, dict):
                gaps.extend(_walk(v, full + "."))
            elif v is None:
                gaps.append(full)
        return gaps

    subset = {k: cfg[k] for k in required_keys if k in cfg}
    return _walk(subset)


def _soil_class(value: float, thresholds: dict) -> str:
    """Classify a soil nutrient reading into low / medium / high."""
    if value < thresholds["low_below"]:
        return "low"
    if value > thresholds["high_above"]:
        return "high"
    return "medium"


def _ratio(doses: dict) -> str:
    """Return NPK ratio string relative to the smallest positive dose."""
    positives = [v for v in doses.values() if v > 0]
    if not positives:
        return "0:0:0"
    base = min(positives)
    return ":".join(
        f"{doses[n] / base:.1f}" if doses[n] > 0 else "0.0"
        for n in NUTRIENTS
    )


def _build_schedule(cfg: dict, doses: dict, area_ha: float) -> list[dict]:
    """Expand split_schedule fractions into actual kg/ha and kg/field values."""
    schedule = []
    for split in cfg.get("split_schedule", []):
        stage_doses = {}
        for n in NUTRIENTS:
            frac = split.get(n, 0.0)
            kg_ha = round(doses[n] * frac, 1)
            stage_doses[n] = {
                "fraction": frac,
                "kg_per_ha": kg_ha,
                "kg_field": round(kg_ha * area_ha, 1),
            }
        schedule.append({"stage": split["stage"], "nutrients": stage_doses})
    return schedule


# ─── Dataset-median fallback ──────────────────────────────────────────────────

_MEDIANS: Optional[Dict] = None


def _load_medians() -> Dict:
    """
    Compute per-crop median N, P, K from the training CSV once.
    Used only when the farmer leaves soil values blank (low-confidence fallback).
    """
    global _MEDIANS
    if _MEDIANS is not None:
        return _MEDIANS

    csv_path = _HERE / "data" / "raw" / "crop_data.csv"
    if not csv_path.exists():
        _MEDIANS = {}
        return _MEDIANS

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        _MEDIANS = (
            df.groupby("Crop_Type")[
                ["Nitrogen_mg_kg", "Phosphorus_mg_kg", "Potassium_mg_kg"]
            ]
            .median()
            .rename(columns={
                "Nitrogen_mg_kg": "N",
                "Phosphorus_mg_kg": "P",
                "Potassium_mg_kg": "K",
            })
            .to_dict("index")
        )
    except Exception:
        _MEDIANS = {}

    return _MEDIANS


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_dose(
    crop: str,
    target_yield: float,
    soil: Dict[str, Optional[float]],
    area_ha: float = 1.0,
) -> Dict:
    """
    Compute fertilizer dose for a given crop.

    Parameters
    ----------
    crop         : Crop name exactly as in crop_formulas.json, e.g. "Rice"
    target_yield : Desired yield in tonnes per hectare
    soil         : {"N": float|None, "P": float|None, "K": float|None} in mg/kg
    area_ha      : Field area in hectares (used only for schedule totals)

    Returns
    -------
    dict with keys:
        status, method, confidence, source,
        N_kg_ha, P2O5_kg_ha, K2O_kg_ha, npk_ratio,
        schedule, trace, [missing]
    """
    trace: list[str] = []
    confidence = "high"

    # ── 1. Lookup crop config ──────────────────────────────────────────────
    cfg = _CFG.get(crop)
    if cfg is None:
        supported = [k for k in _CFG if not k.startswith("_")]
        return {
            "status": "unsupported_crop",
            "crop": crop,
            "supported_crops": supported,
            "trace": [f"Crop '{crop}' not found in crop_formulas.json"],
        }

    trace.append(f"Crop: {crop} | method: {cfg['method']} | verified: {cfg.get('verified', False)}")

    # ── 2. Check completeness ─────────────────────────────────────────────
    gaps = _find_missing(cfg)
    if gaps:
        return {
            "status": "config_incomplete",
            "crop": crop,
            "missing": gaps,
            "trace": [f"Config incomplete. Fix before computing: {gaps}"],
        }

    # ── 3. Validate target yield ──────────────────────────────────────────
    if target_yield <= 0:
        return {
            "status": "invalid_input",
            "error": "target_yield must be > 0",
            "trace": ["Invalid target_yield"],
        }

    # ── 4. Handle missing soil values (dataset-median fallback) ───────────
    soil_used = dict(soil)
    medians = _load_medians()
    for key in ["N", "P", "K"]:
        if soil_used.get(key) is None:
            crop_medians = medians.get(crop, {})
            fallback = crop_medians.get(key)
            if fallback is not None:
                soil_used[key] = fallback
                confidence = "low"
                trace.append(
                    f"Soil {key} missing → using dataset median {fallback:.1f} mg/kg (low confidence)"
                )
            else:
                return {
                    "status": "missing_soil_data",
                    "error": f"Soil {key} is required but not provided and no dataset fallback available.",
                    "trace": trace,
                }

    # ── 5. Compute doses ──────────────────────────────────────────────────
    method = cfg["method"]
    doses: Dict[str, float] = {}

    if method == "stcr":
        # F = a * T - b * S
        stcr = cfg["stcr"]
        for n in NUTRIENTS:
            s_key = _SOIL_KEY[n]
            raw = stcr[n]["a"] * target_yield - stcr[n]["b"] * soil_used[s_key]
            doses[n] = max(0.0, raw)
            trace.append(
                f"STCR {n}: {stcr[n]['a']}×{target_yield} - {stcr[n]['b']}×{soil_used[s_key]:.1f} = {raw:.1f}"
            )
        if confidence == "high":
            confidence = "high"

    else:
        # Nutrient-balance: F = (T × uptake - supply) / efficiency
        for n in NUTRIENTS:
            s_key = _SOIL_KEY[n]
            uptake = cfg["uptake_kg_per_tonne"][n]
            eff = cfg["efficiency"][n]
            cls = _soil_class(soil_used[s_key], cfg["soil_class_thresholds"][s_key])
            supply_frac = cfg["soil_supply_fraction"][cls]
            total_need = target_yield * uptake
            soil_supply = total_need * supply_frac
            raw_dose = (total_need - soil_supply) / eff
            doses[n] = max(0.0, raw_dose)
            trace.append(
                f"{n}: need={total_need:.1f} soil_class={cls} supply_frac={supply_frac} "
                f"supply={soil_supply:.1f} eff={eff} → raw={raw_dose:.1f}"
            )
        if confidence == "high":
            confidence = "medium"   # nutrient_balance is inherently an estimate

    # ── 6. Apply max dose caps ────────────────────────────────────────────
    caps = cfg["max_dose_kg_ha"]
    for n in NUTRIENTS:
        if doses[n] > caps[n]:
            trace.append(f"{n} capped from {doses[n]:.1f} to {caps[n]}")
            doses[n] = caps[n]

    # ── 7. Round ──────────────────────────────────────────────────────────
    doses = {n: round(doses[n], 1) for n in NUTRIENTS}

    # ── 8. Build schedule ─────────────────────────────────────────────────
    schedule = _build_schedule(cfg, doses, area_ha)

    # ── 9. Assemble result ────────────────────────────────────────────────
    return {
        "status": "ok",
        "crop": crop,
        "method": method,
        "confidence": confidence,
        "verified": cfg.get("verified", False),
        "source": cfg.get("source", ""),
        "target_yield_t_ha": target_yield,
        "area_ha": area_ha,
        "soil_input_mg_kg": soil_used,
        "N_kg_ha": doses["N"],
        "P2O5_kg_ha": doses["P2O5"],
        "K2O_kg_ha": doses["K2O"],
        "N_field_kg": round(doses["N"] * area_ha, 1),
        "P2O5_field_kg": round(doses["P2O5"] * area_ha, 1),
        "K2O_field_kg": round(doses["K2O"] * area_ha, 1),
        "npk_ratio": _ratio(doses),
        "schedule": schedule,
        "trace": trace,
    }


def supported_crops() -> list[str]:
    """Return the list of crops that have config entries."""
    return [k for k in _CFG if not k.startswith("_")]


def get_crop_config(crop: str) -> Optional[dict]:
    """Return raw config for a crop (or None if not found)."""
    return _CFG.get(crop)
