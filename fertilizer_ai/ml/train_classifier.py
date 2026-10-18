"""
ml/train_classifier.py
────────────────────────
Trains the XGBoost label cross-check classifier.

Steps:
  1. Load CSV
  2. Select only the features defined in ml/features.py (no leakage)
  3. Impute missing optional columns with median/mode
  4. Encode categoricals with OrdinalEncoder (unknown → -1)
  5. Train XGBClassifier
  6. Report classification report + stratified 5-fold macro-F1
  7. Save model to ml/fert_classifier.joblib

Run:  python -m ml.train_classifier
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent.parent
sys.path.insert(0, str(_HERE))

from ml.features import CAT, FEATURES, NUM, TARGET  # noqa

# ─── Load data ────────────────────────────────────────────────────────────────
CSV_PATH = _HERE / "data" / "raw" / "crop_data.csv"
df = pd.read_csv(CSV_PATH)

print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} cols")
print(f"Target distribution:\n{df[TARGET].value_counts()}\n")

# Encode target
le = LabelEncoder()
y = le.fit_transform(df[TARGET])
print(f"Classes: {list(le.classes_)}\n")

X = df[FEATURES].copy()

# ─── Preprocessing pipeline ───────────────────────────────────────────────────
# Categoricals: OrdinalEncoder (handles unseen values at inference)
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )),
])

# Numerics: median imputation (handles optional columns that may be NaN at inference)
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
])

pre = ColumnTransformer([
    ("cat", cat_pipe, CAT),
    ("num", num_pipe, NUM),
])

# ─── Model ────────────────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
except ImportError:
    print("XGBoost not installed. Falling back to RandomForest.")
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

pipeline = Pipeline([("pre", pre), ("clf", clf)])

# ─── Train / test split ───────────────────────────────────────────────────────
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print("Training...")
pipeline.fit(Xtr, ytr)

# ─── Evaluation ───────────────────────────────────────────────────────────────
ypred = pipeline.predict(Xte)
print("\n=== Test Set Report ===")
print(classification_report(yte, ypred, target_names=le.classes_))

cv = cross_val_score(
    pipeline, X, y,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="f1_macro",
    n_jobs=-1,
)
print(f"5-fold CV macro-F1: {cv.mean():.4f} ± {cv.std():.4f}")

# ─── Save ─────────────────────────────────────────────────────────────────────
from ml.bundle import FertilizerModelBundle
OUT_PATH = _HERE / "ml" / "fert_classifier.joblib"
bundle = FertilizerModelBundle(pipeline=pipeline, classes=le.classes_)
joblib.dump(bundle, OUT_PATH)
print(f"\nModel saved -> {OUT_PATH}")
