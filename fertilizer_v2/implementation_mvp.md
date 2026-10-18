# Precision Fertilizer Recommendation: Final MVP Plan

Scope: 8 crops (Tomato, Cotton, Sugarcane, Maize, Potato, Wheat, Groundnut, Rice), Tamil Nadu focus kept as a note only, tabular data only, no image model.

---

## 1. What changed from the first plan

| # | Old plan | Final MVP |
|---|---|---|
| 1 | One global STCR formula | **One engine, one formula per crop**, stored in `config/crop_formulas.json` |
| 2 | Hardcoded rice equations | No coefficients in code. Config only, each with a `source` and a `verified` flag |
| 3 | Dose from ML (XGBoost) | **Dose from the crop formula.** ML only predicts the `Fertilizer_Recommendation` label as a cross-check |
| 4 | Visual Engine with a CNN or YOLO | **Crop Health Proxy** from tabular indices (chlorophyll, stress, pest and disease) |
| 5 | ML trained on all 44 columns | ML trained only on columns **available at prediction time** (see section 3) |
| 6 | NPK ratio as a target | Ratio is **derived** from the three doses, never predicted |
| 7 | SoilGrids fallback | Farmer enters soil values. If missing, use a low-confidence estimate from dataset medians per crop |
| 8 | Every output assumed trustworthy | Every output carries `method`, `confidence` and `trace` |

---

## 2. MVP architecture

```
Farmer input form
      |
      +--> Soil/Nutrient Engine   (per-crop formula -> N, P2O5, K2O kg/ha)
      +--> Crop Health Proxy      (tabular indices -> flags)
      +--> Weather Engine         (Open-Meteo -> rain, temp, ET)
      |
      v
Decision Engine  (rules + ML label cross-check + confidence)
      |
      v
Fertilizer Calculator (urea/DAP/MOP/SSP, field total, bags, split schedule)
      |
      v
JSON response + farmer-readable advisory
```

Folder layout:

```
fertilizer_ai/
├── data/raw/crop_data.csv
├── config/
│   ├── crop_formulas.json
│   ├── fertilizer_products.json
│   └── rules.yaml
├── engines/ (nutrient_engine.py, health_engine.py, weather_engine.py)
├── recommendation/ (decision_engine.py, fertilizer_calculator.py)
├── ml/ (train_classifier.py, features.py)
├── scripts/audit_dataset.py
├── tests/
└── app.py
```

---

## 3. Dataset columns and what each one is used for

The farmer cannot provide things like Leaf Area Index or Chlorophyll Index. So the ML model must only use fields the system actually has at prediction time.

**A. Farmer inputs (required in the form)**
`Crop_Type`, `Soil_Type`, `Growth_Stage`, `Irrigation_Type`, `Nitrogen_mg_kg`, `Phosphorus_mg_kg`, `Potassium_mg_kg`, `Soil_pH`, `Field_Area_Hectares`, `Target_Yield_ton_per_Hectare`

**B. Farmer inputs (optional, improve confidence)**
`Soil_Organic_Matter_Percent`, `Electrical_Conductivity_dS_m`, `Previous_Yield_ton_per_Hectare`, `Crop_Age_Days`, `Plant_Height_cm`, location (lat/lon)

**C. Auto-filled from the weather API**
`Air_Temperature_C`, `Relative_Humidity_Percent`, `Rainfall_mm`, `Wind_Speed_kmph`, `Solar_Radiation_MJ_m2`, `Evapotranspiration_mm`, `Soil_Moisture_Percent` (estimate)

**D. Used only by the Crop Health Proxy (optional input)**
`Chlorophyll_Index`, `Crop_Stress_Index`, `Pest_Pressure_Index`, `Disease_Pressure_Index`, `Weed_Pressure_Index`, `Leaf_Area_Index`

**E. Leakage suspects, excluded from ML until tested**
`Fertilizer_Response_Rate_Percent`, `Nutrient_Uptake_Efficiency_Percent`, `Previous_Fertilizer_Usage_kg_per_Hectare`

**F. Not used in the MVP**
`Soil_Salinity_dS_m` (near-duplicate of EC, check first), `Soil_Density_g_cm3`, `Water_Holding_Capacity_Percent`, `Root_Depth_cm`, `Plant_Population_per_Hectare`, `Water_Usage_Liters_per_Day`, `Drainage_Quality_Index`, `Irrigation_Frequency_per_Week`, `Water_Availability_Percent`, `Sunshine_Hours`, `Soil_Temperature_C`, `Crop_Growth_Rate_cm_day`

---

## 4. Per-crop formula

### 4.1 Config: `config/crop_formulas.json`

Every `null` is a value **you must provide**. Do not guess them.

```json
{
  "Rice": {
    "method": "nutrient_balance",
    "yield_unit": "t_per_ha",
    "uptake_kg_per_tonne": {"N": null, "P2O5": null, "K2O": null},
    "efficiency":          {"N": null, "P2O5": null, "K2O": null},
    "soil_supply_fraction": {"low": null, "medium": null, "high": null},
    "soil_class_thresholds": {
      "N":   {"low_below": null, "high_above": null},
      "P":   {"low_below": null, "high_above": null},
      "K":   {"low_below": null, "high_above": null}
    },
    "max_dose_kg_ha": {"N": null, "P2O5": null, "K2O": null},
    "split_schedule": [
      {"stage": "basal", "N": null, "P2O5": null, "K2O": null},
      {"stage": "top_dress_1", "N": null, "P2O5": null, "K2O": null}
    ],
    "source": "",
    "verified": false
  },
  "Maize": {}, "Cotton": {}, "Sugarcane": {}, "Groundnut": {},
  "Tomato": {}, "Potato": {}, "Wheat": {}
}
```

When a verified STCR equation becomes available for a crop, switch it to:

```json
"method": "stcr",
"stcr": {"N": {"a": 0, "b": 0}, "P2O5": {"a": 0, "b": 0}, "K2O": {"a": 0, "b": 0}}
```
and the engine uses `F = a*T - b*S` for that crop only.

### 4.2 Engine: `engines/nutrient_engine.py`

```python
import json

CFG = json.load(open("config/crop_formulas.json"))
NUT = ["N", "P2O5", "K2O"]
SOIL_KEY = {"N": "N", "P2O5": "P", "K2O": "K"}

def _missing(cfg):
    def walk(d, path=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from walk(v, f"{path}{k}.")
            elif v is None:
                yield path + k
    return list(walk({k: cfg[k] for k in
        ("uptake_kg_per_tonne", "efficiency", "soil_supply_fraction",
         "soil_class_thresholds", "max_dose_kg_ha") if k in cfg}))

def soil_class(value, th):
    if value < th["low_below"]:  return "low"
    if value > th["high_above"]: return "high"
    return "medium"

def compute_dose(crop, target_yield, soil):
    """soil = {"N": value, "P": value, "K": value} in the units used by thresholds."""
    cfg = CFG.get(crop)
    if not cfg:
        return {"status": "unsupported_crop"}
    gaps = _missing(cfg)
    if gaps:
        return {"status": "config_incomplete", "missing": gaps}

    if cfg["method"] == "stcr":
        out = {n: max(0.0, cfg["stcr"][n]["a"] * target_yield
                      - cfg["stcr"][n]["b"] * soil[SOIL_KEY[n]]) for n in NUT}
        method, conf = "stcr", "high"
    else:
        out = {}
        for n in NUT:
            need = target_yield * cfg["uptake_kg_per_tonne"][n]
            cls = soil_class(soil[SOIL_KEY[n]], cfg["soil_class_thresholds"][SOIL_KEY[n]])
            supply = need * cfg["soil_supply_fraction"][cls]
            out[n] = (need - supply) / cfg["efficiency"][n]
        method, conf = "nutrient_balance_estimate", "medium"

    out = {n: round(min(max(v, 0.0), cfg["max_dose_kg_ha"][n]), 1) for n, v in out.items()}
    return {"status": "ok", "method": method, "confidence": conf,
            "N_kg_ha": out["N"], "P2O5_kg_ha": out["P2O5"], "K2O_kg_ha": out["K2O"],
            "npk_ratio": ratio(out), "source": cfg["source"]}

def ratio(d):
    base = min(v for v in d.values() if v > 0) if any(v > 0 for v in d.values()) else 1
    return ":".join(f"{d[n] / base:.1f}" for n in NUT)
```

Rules built in: the engine refuses to compute if any config value is missing, clamps at 0 and at the cap, and always reports the method used.

---

## 5. Fertilizer calculator

`config/fertilizer_products.json` (check grades and bag sizes on real labels):

```json
{
  "urea": {"N": 46, "P2O5": 0,  "K2O": 0,  "bag_kg": 50},
  "dap":  {"N": 18, "P2O5": 46, "K2O": 0,  "bag_kg": 50},
  "mop":  {"N": 0,  "P2O5": 0,  "K2O": 60, "bag_kg": 50},
  "ssp":  {"N": 0,  "P2O5": 16, "K2O": 0,  "bag_kg": 50}
}
```

```python
import json
P = json.load(open("config/fertilizer_products.json"))

def calc_products(n, p, k, area_ha, p_source="dap"):
    notes = []
    src = P[p_source]
    p_kg = p / (src["P2O5"] / 100) if p > 0 else 0
    n_from_p = p_kg * src["N"] / 100
    if n > 0 and n_from_p > n:
        notes.append("DAP supplies more N than required; switched to SSP")
        p_source, src = "ssp", P["ssp"]
        p_kg, n_from_p = p / (src["P2O5"] / 100), 0
    urea = max(0.0, n - n_from_p) / (P["urea"]["N"] / 100)
    mop = k / (P["mop"]["K2O"] / 100) if k > 0 else 0
    out = {}
    for name, per_ha in {p_source: p_kg, "urea": urea, "mop": mop}.items():
        total = per_ha * area_ha
        out[name] = {"kg_per_ha": round(per_ha, 1), "total_kg": round(total, 1),
                     "bags": int(-(-total // P[name]["bag_kg"]))}
    return {"products": out, "notes": notes}
```

---

## 6. Crop Health Proxy (replaces the Visual Engine)

Compute per-crop, per-stage reference percentiles from the CSV, then flag:

```python
import pandas as pd

def build_reference(df):
    g = df.groupby(["Crop_Type", "Growth_Stage"])
    return pd.DataFrame({
        "chl_p10": g.Chlorophyll_Index.quantile(.10),
        "chl_p25": g.Chlorophyll_Index.quantile(.25),
        "stress_p90": g.Crop_Stress_Index.quantile(.90),
        "pest_p90": g.Pest_Pressure_Index.quantile(.90),
        "disease_p90": g.Disease_Pressure_Index.quantile(.90),
    }).to_dict("index")

def health_flags(crop, stage, chl, stress, pest, disease, ref):
    r = ref[(crop, stage)]
    f = []
    if chl is not None and chl < r["chl_p25"]:
        f.append({"issue": "possible_nutrient_stress",
                  "severity": "high" if chl < r["chl_p10"] else "moderate",
                  "next_step": "confirm with soil test; do not raise dose automatically"})
    if pest is not None and pest > r["pest_p90"]:
        f.append({"issue": "pest_pressure", "severity": "high", "next_step": "field scouting"})
    if disease is not None and disease > r["disease_p90"]:
        f.append({"issue": "disease_pressure", "severity": "high", "next_step": "expert inspection"})
    if stress is not None and stress > r["stress_p90"]:
        f.append({"issue": "general_stress", "severity": "moderate", "next_step": "check water and salinity"})
    return f or [{"issue": "none_detected", "severity": "none"}]
```

If these inputs are not provided, skip the engine and report `health: "not_assessed"`.

---

## 7. Weather Engine

```python
import requests, requests_cache
S = requests_cache.CachedSession("cache/weather", expire_after=3600)

def get_weather(lat, lon):
    r = S.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon, "forecast_days": 7, "timezone": "auto",
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,"
                 "et0_fao_evapotranspiration"}, timeout=15)
    r.raise_for_status()
    d = r.json()["daily"]
    rain = [x or 0 for x in d["precipitation_sum"]]
    return {"source": "open-meteo",
            "rain_3d_mm": sum(rain[:3]), "rain_7d_mm": sum(rain[:7]),
            "tmax_3d": max(x for x in d["temperature_2m_max"][:3] if x is not None),
            "missing": any(x is None for x in d["precipitation_sum"][:7])}
```

If the call fails, return `{"source": "unavailable"}` and tell the farmer to check the local forecast.

`config/rules.yaml` holds your thresholds, for example the rain amount that delays top-dressing. **You must supply or approve these numbers.**

---

## 8. ML label cross-check

```python
import pandas as pd, joblib
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

df = pd.read_csv("data/raw/crop_data.csv")
y = LabelEncoder().fit_transform(df.pop("Fertilizer_Recommendation"))

FEATURES = ["Crop_Type", "Soil_Type", "Growth_Stage", "Irrigation_Type",
            "Nitrogen_mg_kg", "Phosphorus_mg_kg", "Potassium_mg_kg", "Soil_pH",
            "Soil_Organic_Matter_Percent", "Electrical_Conductivity_dS_m",
            "Target_Yield_ton_per_Hectare", "Previous_Yield_ton_per_Hectare",
            "Crop_Age_Days", "Air_Temperature_C", "Relative_Humidity_Percent",
            "Rainfall_mm"]
CAT = ["Crop_Type", "Soil_Type", "Growth_Stage", "Irrigation_Type"]
X = df[FEATURES]

pre = ColumnTransformer([("c", OrdinalEncoder(handle_unknown="use_encoded_value",
                                               unknown_value=-1), CAT)],
                        remainder="passthrough")
m = Pipeline([("pre", pre), ("clf", XGBClassifier(n_estimators=300, max_depth=6,
                                                   learning_rate=0.05))])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
m.fit(Xtr, ytr)
print(classification_report(yte, m.predict(Xte)))
print("CV macro-F1:", cross_val_score(m, X, y, cv=StratifiedKFold(5, shuffle=True,
      random_state=42), scoring="f1_macro").mean())
joblib.dump(m, "ml/fert_classifier.joblib")
```

Use: if the ML label disagrees strongly with the formula's dominant nutrient, lower confidence and add a note. Never let ML override the formula dose.

---

## 9. Decision Engine rules

1. Run the nutrient engine. If status is not `ok`, return the missing fields instead of a dose.
2. Run the health proxy. A nutrient-stress flag **never raises the dose**; it adds "retest soil" advice.
3. Run weather. Heavy rain forecast delays top-dressing (threshold from `rules.yaml`). Missing weather adds a caution.
4. Yield check: if target > 1.3 x previous yield, warn (the 1.3 factor is editable in `rules.yaml`).
5. Compute products, field totals and bags.
6. Final confidence = lowest confidence among engines.
7. Always return `method`, `confidence`, `trace`, `source`.

Response shape:

```json
{
  "crop": "Rice",
  "nutrients_kg_ha": {"N": 0, "P2O5": 0, "K2O": 0},
  "npk_ratio": "x:y:z",
  "method": "nutrient_balance_estimate",
  "confidence": "medium",
  "products": {"urea": {}, "dap": {}, "mop": {}},
  "schedule": [],
  "health": [],
  "weather_advice": [],
  "warnings": [],
  "trace": []
}
```

---

## 10. API

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class Req(BaseModel):
    crop: str
    growth_stage: str
    soil_type: str
    irrigation_type: str
    area_ha: float
    target_yield: float
    soil_n: float
    soil_p: float
    soil_k: float
    soil_ph: float
    lat: Optional[float] = None
    lon: Optional[float] = None
    previous_yield: Optional[float] = None
    organic_matter: Optional[float] = None
    ec: Optional[float] = None
    chlorophyll: Optional[float] = None
    stress: Optional[float] = None
    pest: Optional[float] = None
    disease: Optional[float] = None

@app.post("/recommend")
def recommend(r: Req):
    ...
```

Run: `uvicorn app:app --reload`

---

## 11. What is missing and what YOU must provide

### Must have (blocks the MVP)

| # | Item | Why |
|---|---|---|
| 1 | **Unique values and counts of `Fertilizer_Recommendation`** | Defines what the ML target means |
| 2 | **Source of the CSV** (real survey, synthetic, scraped?) | Decides how much to trust ML metrics |
| 3 | **Units and method of N, P, K columns** (mg/kg? which extraction?) | Needed to set soil thresholds and any kg/ha conversion |
| 4 | **Per crop: uptake per tonne** of N, P2O5, K2O (8 crops x 3) | Core of the formula |
| 5 | **Per crop: efficiency** for N, P2O5, K2O | Core of the formula |
| 6 | **Per crop: soil low/medium/high thresholds** for N, P, K (or approve "use CSV quantiles") | Chooses the soil supply fraction |
| 7 | **Soil supply fraction** for low, medium, high | Adjusts requirement for soil fertility |
| 8 | **Per crop: max dose caps** | Safety limit |
| 9 | **Per crop: split schedule** (basal and top-dress percentages) | Application advice |

### Should have (improves quality)

| # | Item |
|---|---|
| 10 | Any verified STCR equations (crop, soil, season, source) |
| 11 | Fertilizer products you want, with bag sizes and prices |
| 12 | Rain and temperature rules for delaying fertilizer, per crop |
| 13 | Whether you want Tamil text in the output |
| 14 | Front end choice: API only, or also Streamlit or web |

### Decisions to confirm

- Keep all 8 crops, or start with 3 to 4 (suggest the crops you can source agronomy numbers for first).
- Health inputs (chlorophyll, stress, pest, disease): optional form fields, or drop the health engine for now.
- Use CSV quantiles for soil classes if you don't have official thresholds (acceptable for MVP, label it as such).

---

## 12. Build order

| Step | Task | Done when |
|---|---|---|
| 1 | Run `audit_dataset.py`, fill the data dictionary | Items 1 to 3 answered |
| 2 | Fill `crop_formulas.json` for one crop | Engine returns `status: ok` |
| 3 | Fertilizer calculator + unit tests | Hand-checked numbers match |
| 4 | Repeat step 2 for the other crops | All crops `verified` or flagged |
| 5 | Health proxy + weather engine | Both return valid JSON |
| 6 | ML cross-check, leakage test, macro-F1 | Metrics reported honestly |
| 7 | Decision engine + API | End-to-end request works |
| 8 | Expert review of outputs | Sample recommendations signed off |

## 13. Tests to pass before calling it an MVP

- A hand-calculated dose for each crop matches the engine output.
- Missing config values return `config_incomplete`, never a number.
- Output is never negative and never above the cap.
- DAP excess-N case switches product.
- Bags are rounded up from the field total.
- Weather API failure still returns a valid response.
- Unseen categories do not crash the ML model.
- Every response includes `method`, `confidence` and `trace`.

## 14. How to describe it honestly

- "Per-crop nutrient-balance recommendations, with crop-specific STCR equations where verified."
- Nutrient-balance outputs are estimates, not STCR.
- ML is a consistency check, not the source of the dose.
- Gains in yield and cost need field trials; the MVP does not prove them.
