# AI-Powered Precision Fertilizer & Crop Health Advisory System

An end-to-end intelligent agronomy platform that provides precision fertilizer dosage recommendations, real commercial bag calculations (DAP, Urea, MOP), live weather spray safety analysis, and offline computer vision leaf diagnosis.

---

## 🌟 Key Features

1. **Leaf Doctor (Computer Vision)**
   - Pretrained **MobileNetV2** model fine-tuned on 38 PlantVillage crop disease classes.
   - **100% Offline & Local**: Model weights (`pytorch_model.bin`, 9.3 MB) are included locally in `fertilizer_ai/vision/models/mobilenet_v2_plant_disease/`. No cloud API required.
   - Directly adjusts nutrient dosage based on diagnosis (e.g., boosts $K_2O$ by +25% for leaf scorch, restricts foliar Nitrogen by -15% for fungal blight to prevent pathogen proliferation).

2. **Agronomy Decision Engine (STCR / ICAR)**
   - Calculates field-specific $N$, $P_2O_5$, $K_2O$ requirements using Soil Test Crop Response (STCR) equations.
   - Converts scientific nutrient requirements into real commercial market bags (DAP 50kg, Urea 45kg, MOP 50kg) with estimated costs in ₹ INR.
   - Produces a staged split application timeline (Basal, Vegetative top-dress, Panicle/Flowering).

3. **Machine Learning Cross-Check (XGBoost)**
   - Trained on 7,000 multi-soil trial records (`fertilizer_ai/ml/fert_classifier.joblib`, 3.9 MB) to validate formula consistency.

4. **Live Meteorological Risk Scoring**
   - Integrates live temperature, humidity, wind, and 48-hour rainfall forecast to compute a Spray Safety Score (0-100) and warn against nutrient leaching.

5. **Clean Interfaces**
   - **Farmer Web Interface**: Pure vanilla HTML/CSS/JS interface served directly by FastAPI (`http://localhost:8000/`), no React build required.
   - **Desktop App**: TypeScript/Electron application in `uv-code/`.

---

## 📁 Repository Structure

```text
UV/
├── fertilizer_ai/
│   ├── app.py                      # FastAPI REST application & static file server
│   ├── static/
│   │   └── index.html              # Farmer-facing HTML dashboard
│   ├── vision/
│   │   ├── deficiency_engine.py    # Leaf scan inference pipeline (PyTorch)
│   │   └── models/
│   │       └── mobilenet_v2_plant_disease/ # Local HuggingFace model weights (9.3 MB)
│   ├── recommendation/
│   │   ├── decision_engine.py      # Core orchestration & leaf dose adjustment
│   │   ├── fertilizer_calculator.py# DAP/Urea/MOP bag algebra
│   │   └── stcr_engine.py          # ICAR STCR equation solvers
│   ├── engines/
│   │   ├── nutrient_engine.py      # Nutrient calculations
│   │   └── weather_engine.py       # Weather & rain safety scoring
│   ├── ml/
│   │   ├── fert_classifier.joblib  # Trained XGBoost model (3.9 MB)
│   │   └── bundle.py               # ML loading bundle
│   ├── llm/
│   │   └── advisory_engine.py      # Advisory generator (with Leaf Diagnosis section)
│   └── tests/                      # Automated test suite (pytest)
├── uv-code/                        # Electron / TypeScript desktop application
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation
```

---

## 🚀 Getting Started

### 1. Requirements
- Python 3.10+
- PyTorch, FastAPI, Uvicorn, Scikit-learn, XGBoost, Pillow, NumPy

### 2. Install Dependencies
```bash
cd fertilizer_ai
pip install -r requirements.txt
```

### 3. Run the Backend & Web Interface
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser and navigate to:
👉 **http://localhost:8000/**

### 4. Run Automated Tests
```bash
pytest tests/
```
