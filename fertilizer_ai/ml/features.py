"""
ml/features.py
──────────────
Single source of truth for which columns the ML classifier uses.
Both training and inference must import from here.

Only includes columns available at prediction time (no leakage suspects).
"""

FEATURES = [
    # Farmer required
    "Crop_Type",
    "Soil_Type",
    "Growth_Stage",
    "Irrigation_Type",
    "Nitrogen_mg_kg",
    "Phosphorus_mg_kg",
    "Potassium_mg_kg",
    "Soil_pH",
    "Target_Yield_ton_per_Hectare",
    # Farmer optional
    "Soil_Organic_Matter_Percent",
    "Electrical_Conductivity_dS_m",
    "Previous_Yield_ton_per_Hectare",
    "Crop_Age_Days",
    # Auto-filled from weather API
    "Air_Temperature_C",
    "Relative_Humidity_Percent",
    "Rainfall_mm",
]

# Categorical columns (need encoding)
CAT = ["Crop_Type", "Soil_Type", "Growth_Stage", "Irrigation_Type"]

# Numeric columns (used as-is, allow NaN → imputed in pipeline)
NUM = [f for f in FEATURES if f not in CAT]

TARGET = "Fertilizer_Recommendation"

# Leakage suspects (excluded from MVP – test before including):
# "Fertilizer_Response_Rate_Percent"
# "Nutrient_Uptake_Efficiency_Percent"
# "Previous_Fertilizer_Usage_kg_per_Hectare"
