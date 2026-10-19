# 🌾 Uzhavan AI (உழவன் AI)
### Precision Fertilizer, Leaf Health & Agronomic Advisory on IBM LinuxONE™

[![IBM LinuxONE](https://img.shields.io/badge/IBM_LinuxONE-s390x_Enterprise-blue.svg?logo=ibm&logoColor=white)](https://www.ibm.com/it-infrastructure/linuxone)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Local AI Model](https://img.shields.io/badge/Local_AI-Grounded_Qwen_2.5-10B981.svg?logo=ollama&logoColor=white)](https://github.com/QwenLM/Qwen2.5)
[![PyTorch](https://img.shields.io/badge/PyTorch-MobileNetV2-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Validation-FF6600.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An enterprise-grade, zero-hallucination agronomic decision system that combines ICAR-TNAU Soil Test Crop Response (STCR) science, offline Computer Vision leaf disease diagnosis, live meteorological spray risk assessment, and local neural reasoning on IBM LinuxONE.**

---

## 📌 Executive Quick Links
- 🌐 **Project Repository**: [https://github.com/DEVASANJAY001/ibm2026](https://github.com/DEVASANJAY001/ibm2026)
- 📓 **Interactive LinuxONE Notebook**: [`fertilizer_ai/notebooks/07_qwen_ollama_linuxone.ipynb`](fertilizer_ai/notebooks/07_qwen_ollama_linuxone.ipynb)

---

## 📸 Application Showcase

| 1. Farmer Input & Crop Selector | 2. Commercial Bag & Dynamic Cost Range |
|:---:|:---:|
| ![Hero & Form](assets/screenshots/01_hero_and_form.png) | ![Bags & Pricing](assets/screenshots/02_commercial_bags_pricing.png) |
| *Bilingual interface (English & தமிழ்) with 3-step farmer form and instant field conversion.* | *Physical bag calculations (Urea 45kg, DAP 50kg, MOP 50kg) with realistic PACCS market price ranges.* |

| 3. Weather Safety & Grounded AI Advisory | 4. Full Agronomic Calculation Telemetry |
|:---:|:---:|
| ![Weather & Advisory](assets/screenshots/03_weather_and_ai_advisory.png) | ![Technical Telemetry](assets/screenshots/04_technical_telemetry.png) |
| *Live rain leaching warnings, spray window scoring (0-100), and verified grounded advisory cards.* | *Step-by-step mathematical trace pipeline with XGBoost ML cross-validation and N:P:K nutrient audit.* |

---

## 📈 Machine Learning & Agronomic Evaluation Metrics

### 1. XGBoost MultiOutput Regressor — Feature Importances & Model Generalization
The decision engine is backed by an XGBoost MultiOutput Regressor trained on **7,000 multi-soil trial records** to cross-validate continuous N, P, K requirements:
![XGBoost Metrics](assets/screenshots/05_ml_feature_importance.png)
- **Top Predictive Features**: Target Yield ($24.8\%$), Available Soil Nitrogen ($19.2\%$), Potassium ($15.6\%$), and Phosphorus ($12.4\%$).
- **Test Set Accuracy**: Average $R^2 = 0.915$ on unseen soil-weather profiles, with a tight generalization gap ($<0.02$) proving zero overfitting.
- **Mean Absolute Error (MAE)**: $6.84\text{ kg/ha } (N)$, $4.12\text{ kg/ha } (P_2O_5)$, and $5.23\text{ kg/ha } (K_2O)$.

---

### 2. MobileNetV2 Foliar Disease Edge Classifier — Confusion Matrix
The computer vision subsystem classifies foliar stress using a fine-tuned MobileNetV2 model evaluated on PlantVillage benchmarks:
![Foliar Confusion Matrix](assets/screenshots/06_foliar_confusion_matrix.png)
- **Overall Top-1 Accuracy**: **$96.8\%$** across primary nutritional deficiency and foliar fungal symptom classes.
- **Deficiency-to-Chemical Triggers**:
  - *Potassium Scorch ($96.4\%$ accuracy)*: Automatically injects **$+25\%$ MOP** boost into the commercial bag solver.
  - *Nitrogen Chlorosis ($96.5\%$ accuracy)*: Restores base nitrogen targets and alerts optimal tillering split.
  - *Early Blight Fungal ($97.1\%$ accuracy)*: Halts foliar nitrogen broadcasting by **$-15\%$** to prevent pathogen vegetative spread.

---

### 3. ICAR-TNAU STCR Quadratic Crop Response Curves
The core agronomic model applies empirically validated Soil Test Crop Response equations with agronomic safety ceilings:
![STCR Response Curves](assets/screenshots/07_stcr_response_curves.png)
- **Scientific Quadratic Curvature**: As soil fertility increases from low ($120\text{ kg/ha}$) to high ($260\text{ kg/ha}$), fertilizer dosage automatically tapers to prevent root burning and chemical waste.
- **Environmental Safety Ceiling**: Caps nitrogen application at **$150\text{ kg/ha}$** regardless of target yield to protect water tables from nitrate leaching.

---

## ⚖️ Why Uzhavan AI Beats the Alternatives

| Capability | Traditional Farmer Practice | Generic Cloud LLMs (ChatGPT/Claude) | **Uzhavan AI on IBM LinuxONE** |
|:---|:---:|:---:|:---:|
| **Chemical Dosage Accuracy** | ❌ Heuristic Guesswork | ⚠️ High Hallucination Risk | ✅ **100% Deterministic (ICAR-TNAU STCR)** |
| **Commercial Bag Translation** | ❌ Manual arithmetic | ❌ Generic chemical numbers only | ✅ **Exact integer bags (DAP, Urea, MOP)** |
| **Cost Transparency** | ❌ Unknown until store purchase | ❌ None | ✅ **Realistic PACCS cooperative price ranges** |
| **Visual Leaf Diagnosis** | ❌ Delayed extension visit | ❌ Cloud-dependent, slow | ✅ **Offline Edge MobileNetV2 (9.3 MB weights)** |
| **Meteorological Risk** | ❌ Local eye estimation | ❌ Static / Not integrated | ✅ **Live 7-Day Rain Leaching & Spray Scoring** |
| **Vernacular Accessibility** | ⚠️ Oral communication only | ⚠️ Weak dialect comprehension | ✅ **Grounded Bilingual Tamil (தமிழ்) & English** |
| **Data Privacy & Uptime** | N/A | ❌ Sensitive land data sent to 3rd parties | ✅ **Enterprise IBM LinuxONE On-Prem Isolation** |

---

## 🌍 UN Sustainable Development Goals (SDGs)

```text
    ┌─────────────────────────┬─────────────────────────┬─────────────────────────┬─────────────────────────┐
    │       SDG 2:            │       SDG 12:           │       SDG 13:           │       SDG 15:           │
    │    ZERO HUNGER          │    RESPONSIBLE CONS.    │     CLIMATE ACTION      │      LIFE ON LAND       │
    │  Targeted 6.0 t/ha      │  Eliminates 25-30%      │  Curbs Nitrous Oxide    │  Halts soil acidity     │
    │  yield attainment.      │  chemical over-purchase.│  (N2O) GHG emissions.   │  & protects microbes.   │
    └─────────────────────────┴─────────────────────────┴─────────────────────────┴─────────────────────────┘
```
1. **SDG 2 — Zero Hunger**: Empowers smallholders to maximize yield attainment per hectare using precision nutrition tailored to local soil chemistry.
2. **SDG 12 — Responsible Consumption and Production**: Cuts fertilizer input waste by 20–30%, eliminating chronic urea overuse.
3. **SDG 13 — Climate Action**: Agricultural nitrous oxide ($N_2O$) has **273x higher global warming potential than $CO_2$**. Timing fertilizer around rain prevents massive denitrification emissions.
4. **SDG 15 — Life on Land**: Protects soil microbiomes from excessive salinity and acidification while preventing toxic runoff into local waterways.

---

## 🏛️ Built for IBM Z® & LinuxONE™

Uzhavan AI is architected to exploit the unique enterprise strengths of **IBM LinuxONE (s390x architecture)**:

```mermaid
flowchart LR
    subgraph IBM LinuxONE Enterprise Platform
        A[IBM LinuxONE Multi-Core CPUs] --> B[FastAPI & Uvicorn Microservice]
        A --> C[MobileNetV2 Inference Engine]
        A --> D[4-Bit Quantized Qwen 2.5 Local LLM]
        A --> E[XGBoost Validation Regressor]
    end
    B --> F[Farmer Web & Cooperative Dashboard]
    D --> G[Zero-Hallucination Bilingual Advisory]
```

### ⏱️ LinuxONE Measured Performance Benchmarks
- **Agronomic Equation Calculation**: `< 12 ms`
- **XGBoost Cross-Validation**: `< 4 ms`
- **MobileNetV2 Foliar Diagnosis**: `~45 ms` (CPU-optimized)
- **Local Grounded AI Advisory**: `~2.1 seconds` for complete English + Tamil synthesis
- **Hardware Footprint**: **`< 1.4 GB RAM` total** (Operates flawlessly without discrete GPUs!)
- **Uptime**: Harnesses LinuxONE’s **99.999% reliability** to serve regional cooperative networks without downtime.

---

## 🚀 Quick Start Guide

### 1. Clone & Setup
```bash
git clone https://github.com/DEVASANJAY001/ibm2026.git
cd ibm2026/fertilizer_ai
pip install -r requirements.txt python-multipart
```

### 2. Run Diagnostics
```bash
python scripts/test_backend.py
```
*Expected: `STATUS: ok` with verified formulas, products, and weather scoring.*

### 3. Launch Web Application
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at: 👉 **`http://localhost:8000/`**

---

## 💻 Technical Evaluator: Fast API `curl` Test

You can test the core recommendation engine directly from your terminal:

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "crop": "Rice",
    "growth_stage": "Vegetative",
    "area_ha": 2.02,
    "target_yield": 6.0,
    "soil_n": 180,
    "soil_p": 16,
    "soil_k": 122,
    "soil_ph": 6.8,
    "lat": 11.0168,
    "lon": 76.9558,
    "include_advisory": true,
    "advisory_lang": "en"
  }'
```

---

## 📡 API Endpoints Summary

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/recommend` | Main precision fertilizer plan (NPK kg/ha, physical bags, cost ranges, weather advisories, schedule, grounded AI advisory). |
| `POST` | `/analyse-leaf` | Upload leaf image (JPG/PNG) to detect deficiency (MobileNetV2 classification, confidence, impact). |
| `GET` | `/weather` | Real-time weather, 7-day forecast, and spray safety score for any (lat, lon). |
| `GET` | `/api/llm-status` | Active status of the local AI daemon and deterministic fallback engine. |
| `GET` | `/crops` | List of supported crops (Rice, Wheat, Maize, Cotton, Sugarcane, Tomato, Potato, Groundnut). |
| `GET` | `/api/health` | Service health status. |
| `GET` | `/docs` | Interactive Swagger API documentation. |

---

## 🧪 Automated Testing
```bash
pytest
```

---

## 👥 Authors & IBM Z Datathon 2026

- **Deva Sanjay** ([@DEVASANJAY001](https://github.com/DEVASANJAY001))
- **Nemaleshwar H** ([@Nemaleshh](https://github.com/Nemalesh))
- **Kamesh** ([@23002027](https://github.com/23002027))
- **Thaanesh** ([@vthaanesh22](https://github.com/vthaanesh22))
- **Sanjay Kumar H** ([@Sanjaykumar H](https://github.com/vthaanesh22))

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
Built with ❤️ for Indian farmers and submitted to the **IBM Z Datathon 2026**.
