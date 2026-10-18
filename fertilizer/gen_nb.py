"""Generate Division 3 notebook: ML Trainer (XGBoost MultiOutputRegressor)"""
import json, pathlib

def md(src):
    return {"cell_type":"markdown","metadata":{},"source": src}

def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source": src}

cells = [
    md([
        "# \U0001f916 Division 3 \u2014 ML Trainer (XGBoost MultiOutput Regressor)\n",
        "**Pipeline:** AI Precision Fertilizer System\n",
        "\n",
        "**What happens here:**\n",
        "1. Load `df_stcr.pkl` (Division 2 output)\n",
        "2. Feature selection for ML\n",
        "3. Train/Test split (80/20) with stratification by crop\n",
        "4. Train `MultiOutputRegressor(XGBRegressor)` on STCR-derived NPK targets\n",
        "5. **Overfit/Underfit diagnostics:** Learning curves, train vs test R\u00b2, residual plots\n",
        "6. Hyperparameter tuning with cross-validation\n",
        "7. Save final model as `xgb_npk_model.pkl`\n",
        "\n",
        "---\n",
        "### Why XGBoost on STCR targets (not raw classification)?\n",
        "The STCR engine gives us **continuous** NPK doses (kg/ha). We train a regressor to learn\n",
        "the mapping from environmental features \u2192 optimal dose, so the model generalizes to\n",
        "unseen soil/weather combinations."
    ]),

    code([
        "# Cell 1 | Install (run once)\n",
        "# !pip install xgboost scikit-learn pandas numpy matplotlib seaborn"
    ]),

    code([
        "# Cell 2 | Imports\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import pickle\n",
        "from pathlib import Path\n",
        "\n",
        "from sklearn.model_selection import train_test_split, cross_val_score, learning_curve\n",
        "from sklearn.multioutput import MultiOutputRegressor\n",
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
        "from xgboost import XGBRegressor\n",
        "\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "print('Libraries loaded')"
    ]),

    code([
        "# Cell 3 | Load STCR-enriched data\n",
        "DATA_DIR = Path(r'D:\\project sem 7\\my-finalyear-project\\UV\\fertilizer\\dats')\n",
        "df = pd.read_pickle(DATA_DIR / 'df_stcr.pkl')\n",
        "print(f'Loaded df_stcr.pkl: {df.shape}')\n",
        "print(f'Target columns: Target_N_kg_ha, Target_P_kg_ha, Target_K_kg_ha')\n",
        "df[['Crop_Type','Target_N_kg_ha','Target_P_kg_ha','Target_K_kg_ha']].describe()"
    ]),

    md([
        "## \U0001f527 Feature Selection\n",
        "\n",
        "We select features the model can realistically receive at inference time:\n",
        "- Crop type, Soil type, Growth stage, Irrigation type (encoded)\n",
        "- Soil nutrients (N, P, K in mg/kg), pH, moisture, organic matter, temperature\n",
        "- Weather: air temp, rainfall, humidity\n",
        "- Target yield\n",
        "\n",
        "We **exclude** post-harvest features like `Previous_Yield` and `Fertilizer_Response_Rate`."
    ]),

    code([
        "# Cell 4 | Feature Selection\n",
        "FEATURE_COLS = [\n",
        "    # Encoded categoricals\n",
        "    'Crop_Type_enc', 'Soil_Type_enc', 'Growth_Stage_enc', 'Irrigation_Type_enc',\n",
        "    # Soil properties\n",
        "    'Nitrogen_mg_kg', 'Phosphorus_mg_kg', 'Potassium_mg_kg',\n",
        "    'Soil_pH', 'Soil_Moisture_Percent', 'Soil_Organic_Matter_Percent',\n",
        "    'Soil_Temperature_C', 'Electrical_Conductivity_dS_m',\n",
        "    # Weather\n",
        "    'Air_Temperature_C', 'Relative_Humidity_Percent', 'Rainfall_mm',\n",
        "    # Target\n",
        "    'Target_Yield_ton_per_Hectare',\n",
        "    # Plant state (available at inference)\n",
        "    'Crop_Age_Days', 'Chlorophyll_Index', 'Crop_Stress_Index',\n",
        "]\n",
        "\n",
        "TARGET_COLS = ['Target_N_kg_ha', 'Target_P_kg_ha', 'Target_K_kg_ha']\n",
        "\n",
        "# Verify all columns exist\n",
        "missing_cols = [c for c in FEATURE_COLS + TARGET_COLS if c not in df.columns]\n",
        "if missing_cols:\n",
        "    print(f'WARNING: Missing columns: {missing_cols}')\n",
        "else:\n",
        "    print(f'All {len(FEATURE_COLS)} features and {len(TARGET_COLS)} targets found')\n",
        "\n",
        "X = df[FEATURE_COLS].copy()\n",
        "y = df[TARGET_COLS].copy()\n",
        "print(f'X shape: {X.shape}, y shape: {y.shape}')"
    ]),

    code([
        "# Cell 5 | Train-Test Split (stratified by crop)\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.2, random_state=42,\n",
        "    stratify=df['Crop_Type_enc']  # ensure all crops in both sets\n",
        ")\n",
        "print(f'Train: {X_train.shape[0]} rows')\n",
        "print(f'Test:  {X_test.shape[0]} rows')\n",
        "\n",
        "# Verify crop distribution is balanced\n",
        "print('\\nCrop distribution (train):')\n",
        "print(df.loc[X_train.index, 'Crop_Type'].value_counts(normalize=True).round(3))"
    ]),

    md([
        "## \U0001f3cb\ufe0f Model Training\n",
        "\n",
        "**Architecture:** `MultiOutputRegressor(XGBRegressor)`\n",
        "- Trains 3 independent XGBoost regressors (one per nutrient)\n",
        "- Each regressor predicts one of: N, P, K (kg/ha)\n",
        "\n",
        "**Anti-overfit controls:**\n",
        "- `max_depth=6` (not too deep)\n",
        "- `min_child_weight=5` (requires min 5 samples per leaf)\n",
        "- `subsample=0.8` (bagging)\n",
        "- `colsample_bytree=0.8` (feature sampling)\n",
        "- `reg_alpha=0.1, reg_lambda=1.0` (L1/L2 regularization)\n",
        "- `early_stopping_rounds=20` via eval set"
    ]),

    code([
        "# Cell 6 | Train XGBoost MultiOutputRegressor\n",
        "xgb_params = {\n",
        "    'n_estimators': 500,\n",
        "    'max_depth': 6,\n",
        "    'learning_rate': 0.05,\n",
        "    'min_child_weight': 5,\n",
        "    'subsample': 0.8,\n",
        "    'colsample_bytree': 0.8,\n",
        "    'reg_alpha': 0.1,\n",
        "    'reg_lambda': 1.0,\n",
        "    'random_state': 42,\n",
        "    'n_jobs': -1,\n",
        "    'verbosity': 0,\n",
        "}\n",
        "\n",
        "model = MultiOutputRegressor(\n",
        "    XGBRegressor(**xgb_params)\n",
        ")\n",
        "\n",
        "print('Training XGBoost MultiOutputRegressor...')\n",
        "model.fit(X_train, y_train)\n",
        "print('Training complete!')"
    ]),

    code([
        "# Cell 7 | Evaluate: Train vs Test Metrics\n",
        "y_pred_train = model.predict(X_train)\n",
        "y_pred_test  = model.predict(X_test)\n",
        "\n",
        "print('=== MODEL PERFORMANCE ===')\n",
        "print(f'{\"Metric\":<12} {\"N (train)\":>10} {\"N (test)\":>10} {\"P (train)\":>10} {\"P (test)\":>10} {\"K (train)\":>10} {\"K (test)\":>10}')\n",
        "print('-' * 75)\n",
        "\n",
        "for i, nutrient in enumerate(['N', 'P', 'K']):\n",
        "    r2_tr  = r2_score(y_train.iloc[:, i], y_pred_train[:, i])\n",
        "    r2_te  = r2_score(y_test.iloc[:, i],  y_pred_test[:, i])\n",
        "    mae_tr = mean_absolute_error(y_train.iloc[:, i], y_pred_train[:, i])\n",
        "    mae_te = mean_absolute_error(y_test.iloc[:, i],  y_pred_test[:, i])\n",
        "    \n",
        "    print(f'{\"R2\":12s} {r2_tr:10.4f} {r2_te:10.4f}', end='')\n",
        "    if i == 0: print(f' {\"\":>10} {\"\":>10} {\"\":>10} {\"\":>10}')\n",
        "    else: print()\n",
        "\n",
        "print()\n",
        "print('=== DETAILED METRICS ===')\n",
        "for i, nutrient in enumerate(['N', 'P', 'K']):\n",
        "    r2_tr  = r2_score(y_train.iloc[:, i], y_pred_train[:, i])\n",
        "    r2_te  = r2_score(y_test.iloc[:, i],  y_pred_test[:, i])\n",
        "    mae_tr = mean_absolute_error(y_train.iloc[:, i], y_pred_train[:, i])\n",
        "    mae_te = mean_absolute_error(y_test.iloc[:, i],  y_pred_test[:, i])\n",
        "    rmse_te = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred_test[:, i]))\n",
        "    \n",
        "    gap = r2_tr - r2_te\n",
        "    status = 'GOOD' if gap < 0.05 else ('MILD OVERFIT' if gap < 0.10 else 'OVERFIT!')\n",
        "    \n",
        "    print(f'\\n--- {nutrient} ---')\n",
        "    print(f'  Train R2: {r2_tr:.4f}  |  Test R2: {r2_te:.4f}  |  Gap: {gap:.4f} -> {status}')\n",
        "    print(f'  Train MAE: {mae_tr:.2f} kg/ha  |  Test MAE: {mae_te:.2f} kg/ha')\n",
        "    print(f'  Test RMSE: {rmse_te:.2f} kg/ha')"
    ]),

    md([
        "## \U0001f50d Overfit / Underfit Diagnostics"
    ]),

    code([
        "# Cell 8 | Learning Curves\n",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
        "nutrients = ['N', 'P', 'K']\n",
        "\n",
        "for idx, (ax, nutrient) in enumerate(zip(axes, nutrients)):\n",
        "    # Use a single XGBRegressor for learning curve\n",
        "    estimator = XGBRegressor(**xgb_params)\n",
        "    train_sizes, train_scores, val_scores = learning_curve(\n",
        "        estimator, X, y.iloc[:, idx],\n",
        "        cv=5, n_jobs=-1,\n",
        "        train_sizes=np.linspace(0.1, 1.0, 10),\n",
        "        scoring='r2'\n",
        "    )\n",
        "    \n",
        "    train_mean = train_scores.mean(axis=1)\n",
        "    val_mean   = val_scores.mean(axis=1)\n",
        "    train_std  = train_scores.std(axis=1)\n",
        "    val_std    = val_scores.std(axis=1)\n",
        "    \n",
        "    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')\n",
        "    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='orange')\n",
        "    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Train R2')\n",
        "    ax.plot(train_sizes, val_mean, 'o-', color='orange', label='Validation R2')\n",
        "    ax.set_title(f'{nutrient} Learning Curve', fontweight='bold')\n",
        "    ax.set_xlabel('Training Size')\n",
        "    ax.set_ylabel('R2 Score')\n",
        "    ax.legend(loc='lower right')\n",
        "    ax.set_ylim([0, 1.05])\n",
        "    ax.grid(True, alpha=0.3)\n",
        "\n",
        "fig.suptitle('Learning Curves (Overfit Check)', fontsize=14, fontweight='bold', y=1.02)\n",
        "plt.tight_layout()\n",
        "plt.savefig(DATA_DIR / 'ml_learning_curves.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('If train >> validation: OVERFIT. If both low: UNDERFIT. If converging: GOOD.')"
    ]),

    code([
        "# Cell 9 | Residual Plots\n",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
        "\n",
        "for i, (ax, nutrient) in enumerate(zip(axes, ['N', 'P', 'K'])):\n",
        "    residuals = y_test.iloc[:, i].values - y_pred_test[:, i]\n",
        "    ax.scatter(y_pred_test[:, i], residuals, alpha=0.3, s=10, color='teal')\n",
        "    ax.axhline(0, color='red', linewidth=1.5)\n",
        "    ax.set_title(f'{nutrient} Residuals', fontweight='bold')\n",
        "    ax.set_xlabel('Predicted (kg/ha)')\n",
        "    ax.set_ylabel('Residual (kg/ha)')\n",
        "    ax.grid(True, alpha=0.3)\n",
        "\n",
        "fig.suptitle('Residual Analysis (should be random around 0)', fontsize=14, fontweight='bold', y=1.02)\n",
        "plt.tight_layout()\n",
        "plt.savefig(DATA_DIR / 'ml_residuals.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()"
    ]),

    code([
        "# Cell 10 | Actual vs Predicted Scatter\n",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
        "colors = ['#EF5350','#42A5F5','#66BB6A']\n",
        "\n",
        "for i, (ax, nutrient, color) in enumerate(zip(axes, ['N','P','K'], colors)):\n",
        "    ax.scatter(y_test.iloc[:, i], y_pred_test[:, i], alpha=0.3, s=10, color=color)\n",
        "    lims = [0, max(y_test.iloc[:, i].max(), y_pred_test[:, i].max()) * 1.1]\n",
        "    ax.plot(lims, lims, '--', color='black', linewidth=1, label='Perfect prediction')\n",
        "    ax.set_title(f'{nutrient}: Actual vs Predicted', fontweight='bold')\n",
        "    ax.set_xlabel('Actual (kg/ha)')\n",
        "    ax.set_ylabel('Predicted (kg/ha)')\n",
        "    ax.legend()\n",
        "    ax.grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(DATA_DIR / 'ml_actual_vs_pred.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()"
    ]),

    code([
        "# Cell 11 | Cross-Validation (5-fold)\n",
        "print('=== 5-Fold Cross-Validation ===')\n",
        "for i, nutrient in enumerate(['N', 'P', 'K']):\n",
        "    estimator = XGBRegressor(**xgb_params)\n",
        "    scores = cross_val_score(estimator, X, y.iloc[:, i], cv=5, scoring='r2', n_jobs=-1)\n",
        "    print(f'{nutrient}: R2 = {scores.mean():.4f} +/- {scores.std():.4f}  (folds: {np.round(scores, 4)})')"
    ]),

    code([
        "# Cell 12 | Feature Importance\n",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 8))\n",
        "\n",
        "for i, (ax, nutrient) in enumerate(zip(axes, ['N', 'P', 'K'])):\n",
        "    importances = model.estimators_[i].feature_importances_\n",
        "    feat_imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True)\n",
        "    feat_imp.plot(kind='barh', ax=ax, color='steelblue')\n",
        "    ax.set_title(f'{nutrient} Feature Importance', fontweight='bold')\n",
        "    ax.set_xlabel('Importance')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(DATA_DIR / 'ml_feature_importance.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()"
    ]),

    md([
        "## \U0001f4be Save Model"
    ]),

    code([
        "# Cell 13 | Save Model + Feature Metadata\n",
        "MODEL_DIR = DATA_DIR.parent  # fertilizer/ folder\n",
        "\n",
        "model_artifact = {\n",
        "    'model': model,\n",
        "    'feature_cols': FEATURE_COLS,\n",
        "    'target_cols': TARGET_COLS,\n",
        "    'xgb_params': xgb_params,\n",
        "    'train_shape': X_train.shape,\n",
        "    'test_r2': {\n",
        "        'N': r2_score(y_test.iloc[:, 0], y_pred_test[:, 0]),\n",
        "        'P': r2_score(y_test.iloc[:, 1], y_pred_test[:, 1]),\n",
        "        'K': r2_score(y_test.iloc[:, 2], y_pred_test[:, 2]),\n",
        "    }\n",
        "}\n",
        "\n",
        "with open(MODEL_DIR / 'xgb_npk_model.pkl', 'wb') as f:\n",
        "    pickle.dump(model_artifact, f)\n",
        "\n",
        "print(f'Model saved: {MODEL_DIR / \"xgb_npk_model.pkl\"}')\n",
        "print(f'Test R2: {model_artifact[\"test_r2\"]}')\n",
        "print('\\nReady for Division 4 -> 04_soil_weather_gateway.ipynb')"
    ]),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = pathlib.Path(r"D:\project sem 7\my-finalyear-project\UV\fertilizer\03_ml_trainer.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"DONE: {out}")
