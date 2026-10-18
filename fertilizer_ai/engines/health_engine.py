"""
engines/health_engine.py
────────────────────────
Crop Health Proxy Engine.

Replaces the Visual/CNN engine.  Uses tabular indices (Chlorophyll, Stress,
Pest, Disease, Weed) from the dataset to build per-crop per-stage reference
percentiles.  At inference, any index reading is compared against those
percentiles to produce health flags.

If no index values are supplied by the farmer → returns health: "not_assessed".
"""

import json
from pathlib import Path
from typing import Dict, Optional

_HERE = Path(__file__).parent.parent

# ─── Reference table: built once from CSV ─────────────────────────────────────
_REF: Optional[Dict] = None


def _build_reference() -> Dict:
    """
    Load the CSV and compute per-crop × per-stage percentiles for health indices.
    Returns a nested dict keyed by (crop, stage).
    """
    global _REF
    if _REF is not None:
        return _REF

    csv_path = _HERE / "data" / "raw" / "crop_data.csv"
    if not csv_path.exists():
        _REF = {}
        return _REF

    import pandas as pd

    df = pd.read_csv(csv_path)
    required_cols = [
        "Crop_Type", "Growth_Stage",
        "Chlorophyll_Index", "Crop_Stress_Index",
        "Pest_Pressure_Index", "Disease_Pressure_Index",
        "Weed_Pressure_Index",
    ]
    df = df[required_cols].dropna()
    g = df.groupby(["Crop_Type", "Growth_Stage"])

    ref = {}
    for (crop, stage), grp in g:
        ref[(crop, stage)] = {
            "n": len(grp),
            "chl_p10": grp["Chlorophyll_Index"].quantile(0.10),
            "chl_p25": grp["Chlorophyll_Index"].quantile(0.25),
            "stress_p90": grp["Crop_Stress_Index"].quantile(0.90),
            "pest_p90": grp["Pest_Pressure_Index"].quantile(0.90),
            "disease_p90": grp["Disease_Pressure_Index"].quantile(0.90),
            "weed_p90": grp["Weed_Pressure_Index"].quantile(0.90),
        }

    _REF = ref
    return _REF


def health_flags(
    crop: str,
    stage: str,
    chlorophyll: Optional[float] = None,
    stress: Optional[float] = None,
    pest: Optional[float] = None,
    disease: Optional[float] = None,
    weed: Optional[float] = None,
) -> Dict:
    """
    Evaluate health indices against reference percentiles.

    Returns dict:
        {
            "status": "assessed" | "not_assessed" | "no_reference",
            "flags": [...],
            "confidence": "high" | "medium" | "low",
            "trace": [...]
        }
    """
    # Check if any index was provided
    provided = {
        "chlorophyll": chlorophyll,
        "stress": stress,
        "pest": pest,
        "disease": disease,
        "weed": weed,
    }
    if all(v is None for v in provided.values()):
        return {
            "status": "not_assessed",
            "flags": [],
            "confidence": "none",
            "trace": ["No health indices provided by farmer."],
        }

    ref = _build_reference()
    key = (crop, stage)

    if key not in ref:
        return {
            "status": "no_reference",
            "flags": [],
            "confidence": "low",
            "trace": [
                f"No reference data for ({crop}, {stage}). "
                "Cannot evaluate health flags."
            ],
        }

    r = ref[key]
    trace = [f"Reference built from {r['n']} samples for ({crop}, {stage})"]
    flags = []
    confidence = "high" if r["n"] >= 30 else "low"

    # ── Chlorophyll (low = nutrient stress) ──────────────────────────────
    if chlorophyll is not None:
        if chlorophyll < r["chl_p10"]:
            flags.append({
                "issue": "possible_nutrient_deficiency",
                "severity": "high",
                "value": round(chlorophyll, 2),
                "reference_p10": round(r["chl_p10"], 2),
                "next_step": "Retest soil; do NOT raise dose automatically.",
            })
            trace.append(f"Chlorophyll {chlorophyll:.2f} < p10 {r['chl_p10']:.2f} → HIGH deficiency flag")
        elif chlorophyll < r["chl_p25"]:
            flags.append({
                "issue": "possible_nutrient_stress",
                "severity": "moderate",
                "value": round(chlorophyll, 2),
                "reference_p25": round(r["chl_p25"], 2),
                "next_step": "Monitor closely; confirm with soil test.",
            })
            trace.append(f"Chlorophyll {chlorophyll:.2f} < p25 {r['chl_p25']:.2f} → MODERATE stress flag")
        else:
            trace.append(f"Chlorophyll {chlorophyll:.2f} OK (≥ p25 {r['chl_p25']:.2f})")

    # ── Pest pressure (high = problem) ───────────────────────────────────
    if pest is not None:
        if pest > r["pest_p90"]:
            flags.append({
                "issue": "high_pest_pressure",
                "severity": "high",
                "value": round(pest, 2),
                "reference_p90": round(r["pest_p90"], 2),
                "next_step": "Immediate field scouting; consider IPM intervention.",
            })
            trace.append(f"Pest {pest:.2f} > p90 {r['pest_p90']:.2f} → HIGH pest flag")
        else:
            trace.append(f"Pest {pest:.2f} OK (≤ p90 {r['pest_p90']:.2f})")

    # ── Disease pressure (high = problem) ────────────────────────────────
    if disease is not None:
        if disease > r["disease_p90"]:
            flags.append({
                "issue": "high_disease_pressure",
                "severity": "high",
                "value": round(disease, 2),
                "reference_p90": round(r["disease_p90"], 2),
                "next_step": "Expert field inspection; hold top-dress until resolved.",
            })
            trace.append(f"Disease {disease:.2f} > p90 {r['disease_p90']:.2f} → HIGH disease flag")
        else:
            trace.append(f"Disease {disease:.2f} OK")

    # ── General stress (high = problem) ──────────────────────────────────
    if stress is not None:
        if stress > r["stress_p90"]:
            flags.append({
                "issue": "general_crop_stress",
                "severity": "moderate",
                "value": round(stress, 2),
                "reference_p90": round(r["stress_p90"], 2),
                "next_step": "Check soil moisture, salinity and drainage.",
            })
            trace.append(f"Stress {stress:.2f} > p90 {r['stress_p90']:.2f} → general stress flag")
        else:
            trace.append(f"Stress {stress:.2f} OK")

    # ── Weed pressure ────────────────────────────────────────────────────
    if weed is not None:
        if weed > r["weed_p90"]:
            flags.append({
                "issue": "high_weed_pressure",
                "severity": "moderate",
                "value": round(weed, 2),
                "reference_p90": round(r["weed_p90"], 2),
                "next_step": "Weed management before next fertilizer application.",
            })
            trace.append(f"Weed {weed:.2f} > p90 {r['weed_p90']:.2f} → weed flag")

    if not flags:
        flags = [{"issue": "none_detected", "severity": "none"}]
        trace.append("All indices within normal range.")

    return {
        "status": "assessed",
        "flags": flags,
        "confidence": confidence,
        "trace": trace,
    }
