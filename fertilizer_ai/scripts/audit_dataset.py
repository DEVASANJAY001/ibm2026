"""
scripts/audit_dataset.py
─────────────────────────
Run this FIRST before training or configuring formulas.

Prints:
  • Shape, columns, dtypes
  • Missing values
  • Fertilizer_Recommendation value counts
  • Crop_Type distribution
  • Per-crop NPK stats (min, p25, median, p75, max)
  • Soil leakage suspect correlations

Usage:  python scripts/audit_dataset.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

CSV = Path(__file__).parent.parent / "data" / "raw" / "crop_data.csv"
df = pd.read_csv(CSV)

print("=" * 60)
print(f"DATASET AUDIT — {CSV.name}")
print("=" * 60)

print(f"\n▶ Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

print("▶ Missing values:")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("   None — all columns complete.")
else:
    print(missing.to_string())

print("\n▶ Fertilizer_Recommendation distribution:")
print(df["Fertilizer_Recommendation"].value_counts().to_string())

print("\n▶ Crop_Type distribution:")
print(df["Crop_Type"].value_counts().to_string())

print("\n▶ Growth Stage values:", sorted(df["Growth_Stage"].unique()))
print("▶ Soil Type values:   ", sorted(df["Soil_Type"].unique()))
print("▶ Irrigation values:  ", sorted(df["Irrigation_Type"].unique()))

print("\n▶ Per-crop NPK statistics (mg/kg):")
crops = sorted(df["Crop_Type"].unique())
for crop in crops:
    c = df[df["Crop_Type"] == crop]
    print(f"\n  ── {crop} (n={len(c)}) ──")
    for col, key in [
        ("Nitrogen_mg_kg", "N"),
        ("Phosphorus_mg_kg", "P"),
        ("Potassium_mg_kg", "K"),
    ]:
        stats = c[col].describe(percentiles=[0.25, 0.5, 0.75])
        print(
            f"  {key}: min={stats['min']:.1f}  p25={stats['25%']:.1f}  "
            f"median={stats['50%']:.1f}  p75={stats['75%']:.1f}  max={stats['max']:.1f}"
        )
    print(f"  Target yield: min={c['Target_Yield_ton_per_Hectare'].min():.1f}  "
          f"median={c['Target_Yield_ton_per_Hectare'].median():.1f}  "
          f"max={c['Target_Yield_ton_per_Hectare'].max():.1f} t/ha")
    print(f"  Fert labels: {dict(c['Fertilizer_Recommendation'].value_counts())}")

print("\n▶ Leakage suspect correlations with Fertilizer_Recommendation:")
leakage_cols = [
    "Fertilizer_Response_Rate_Percent",
    "Nutrient_Uptake_Efficiency_Percent",
    "Previous_Fertilizer_Usage_kg_per_Hectare",
]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(df["Fertilizer_Recommendation"])
for col in leakage_cols:
    if col in df.columns:
        corr = np.corrcoef(df[col].fillna(0), y_enc)[0, 1]
        print(f"   {col}: r = {corr:.4f}")

print("\n▶ Potential near-duplicate columns:")
print("  Soil_Salinity_dS_m vs Electrical_Conductivity_dS_m")
if "Soil_Salinity_dS_m" in df.columns and "Electrical_Conductivity_dS_m" in df.columns:
    corr = df["Soil_Salinity_dS_m"].corr(df["Electrical_Conductivity_dS_m"])
    print(f"  Pearson correlation: {corr:.4f}")

print("\n" + "=" * 60)
print("Audit complete.")
print("=" * 60)
