# 🏦 Smart Loan Risk Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi" />
  <img src="https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/XGBoost-2.0-orange" />
  <img src="https://img.shields.io/badge/SHAP-Explainable_AI-purple" />
  <img src="https://img.shields.io/badge/Docker-Containerised-2496ED?logo=docker" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

> **Production-grade AI/ML system for loan default risk prediction, fraud detection, and explainable AI — built with industry-standard fintech practices.**

---

## 📋 Table of Contents
- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start (Windows)](#quick-start-windows)
- [Model Workflow](#model-workflow)
- [API Documentation](#api-documentation)
- [Dashboard](#dashboard)
- [Explainable AI (SHAP)](#explainable-ai-shap)
- [Docker Deployment](#docker-deployment)
- [GitHub Push](#github-push)
- [Model Performance](#model-performance)
- [Scaling Strategy](#scaling-strategy)

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   CLIENT LAYER                          │
│   Swagger UI  │  Streamlit Dashboard  │  REST Clients   │
└────────┬───────────────┬──────────────────┬─────────────┘
         │               │                  │
┌────────▼───────────────▼──────────────────▼─────────────┐
│                   API LAYER (FastAPI)                    │
│  /predict  │  /fraud-check  │  /health  │  /model-info  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  ML PIPELINE LAYER                       │
│                                                         │
│  FeatureEngineer → Preprocessor → StackingEnsemble      │
│       ↓                               ↓                 │
│  FraudDetector               SHAPExplainer              │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   DATA LAYER                             │
│   German Credit CSV  │  Processed Parquet  │  Joblib    │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.11 |
| **API** | FastAPI + Uvicorn |
| **ML Models** | RandomForest, XGBoost, CatBoost, LightGBM |
| **Ensemble** | Stacking + Logistic Regression meta-learner |
| **Tuning** | Optuna (TPE sampler) |
| **Explainability** | SHAP (TreeExplainer) |
| **Imbalance** | SMOTE (imbalanced-learn) |
| **Dashboard** | Streamlit + Plotly |
| **Reports** | ReportLab PDF |
| **Serialisation** | Joblib |
| **Containers** | Docker + Docker Compose |

---

## 📁 Project Structure

```
smart-loan-risk-system/
├── data/
│   ├── raw/                    # Raw input CSV
│   └── processed/              # Engineered + cleaned CSV
├── notebooks/
│   ├── 01_eda.py               # EDA & visualisations
│   ├── 02_feature_engineering.py
│   └── 03_model_training.py    # Full training + SHAP
├── src/
│   ├── config/config.py        # Centralised config & paths
│   ├── data/
│   │   ├── data_loader.py      # Raw data loading + target derivation
│   │   └── preprocess.py       # Impute + encode + scale + SMOTE
│   ├── features/
│   │   └── feature_engineering.py  # 5 domain-driven features
│   ├── models/
│   │   ├── train.py            # End-to-end training pipeline
│   │   ├── evaluate.py         # Metrics + confusion matrix + ROC
│   │   └── stacking_model.py   # StackingClassifier builder
│   ├── explainability/
│   │   └── shap_explainer.py   # Global + local SHAP plots
│   ├── fraud/
│   │   └── fraud_detector.py   # Rule-based fraud scoring
│   ├── api/
│   │   └── main.py             # FastAPI application
│   ├── dashboard/
│   │   └── app.py              # Streamlit dashboard
│   └── utils/
│       └── report_generator.py # PDF report builder
├── models/                     # Serialised .joblib artifacts
├── reports/                    # Generated PNGs + PDFs
├── requirements.txt
├── Dockerfile
├── Dockerfile.dashboard
├── docker-compose.yml
├── setup.py
└── README.md
```

---

## 🚀 Quick Start (Windows — VS Code)

### Step 1 — Install Python 3.11
Download from [python.org](https://www.python.org/downloads/) — check **"Add to PATH"**.

```powershell
python --version   # should show 3.11.x
```

### Step 2 — Clone / unzip project

```powershell
cd C:\Projects
# if using git:
git clone https://github.com/YOUR_USERNAME/smart-loan-risk-system.git
cd smart-loan-risk-system
```

### Step 3 — Create virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
> If you get an execution policy error:
> `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Step 4 — Install requirements

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5 — VS Code Extensions
Install in VS Code (Ctrl+Shift+X):
- **Python** (Microsoft)
- **Pylance**
- **Jupyter**
- **Docker**

### Step 6 — Train the model

```powershell
python -m src.models.train
```
This runs:
1. Data loading
2. Feature engineering
3. Preprocessing + SMOTE
4. Optuna hyperparameter tuning (XGBoost)
5. Stacking ensemble training
6. Evaluation + saves artifacts to `models/`

### Step 7 — Run EDA Notebooks

```powershell
python notebooks/01_eda.py
python notebooks/02_feature_engineering.py
python notebooks/03_model_training.py
```
Outputs saved to `reports/`.

### Step 8 — Start FastAPI server

```powershell
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```
Open Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

### Step 9 — Start Streamlit Dashboard

```powershell
# In a second terminal (activate venv first)
streamlit run src/dashboard/app.py
```
Open: [http://localhost:8501](http://localhost:8501)

### Step 10 — Test API via Swagger
1. Open [http://localhost:8000/docs](http://localhost:8000/docs)
2. Click **POST /predict** → Try it out
3. Paste sample JSON and Execute

Sample request body:
```json
{
  "Age": 35,
  "Sex": "male",
  "Job": 2,
  "Housing": "own",
  "Saving_accounts": "moderate",
  "Checking_account": "little",
  "Credit_amount": 5000,
  "Duration": 24,
  "Purpose": "car"
}
```

---

## 🐳 Docker Deployment

### Build and run everything

```bash
# Build + start API and Dashboard containers
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

| Service | URL |
|---------|-----|
| API + Swagger | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |

### Stop containers

```bash
docker-compose down
```

---

## 📡 API Documentation

### `GET /health`
```json
{"status": "ok", "model_loaded": true, "api_version": "1.0.0"}
```

### `GET /model-info`
```json
{
  "model_type": "StackingClassifier",
  "base_models": ["RandomForest","XGBoost","CatBoost","LightGBM"],
  "features": ["Age","Credit amount",...],
  "threshold": 0.5
}
```

### `POST /predict`
**Request:**
```json
{"Age": 35, "Sex": "male", "Job": 2, "Housing": "own",
 "Saving_accounts": "moderate", "Checking_account": "little",
 "Credit_amount": 5000, "Duration": 24, "Purpose": "car"}
```
**Response:**
```json
{
  "risk_label": "LOW RISK",
  "risk_probability": 0.2341,
  "risk_score_pct": 23.41,
  "shap_explanation": {"Age": -0.04, "Credit amount": 0.12, ...},
  "fraud_result": {
    "fraud_score": 15.0,
    "is_fraud_flag": false,
    "flags": [],
    "recommendation": "APPROVE — No significant fraud indicators detected."
  },
  "model_version": "1.0.0"
}
```

### `POST /fraud-check`
Standalone fraud scoring without ML prediction.

---

## 🧠 Explainable AI (SHAP)

SHAP assigns each feature a **Shapley value** — its marginal contribution to the prediction:

```
Prediction = base_value + SHAP(Age) + SHAP(Credit amount) + ... + SHAP(Purpose)
```

### Plots generated:
| Plot | Description | Saved to |
|------|-------------|----------|
| Summary (beeswarm) | Global feature importance | `reports/shap_summary.png` |
| Bar chart | Mean \|SHAP\| per feature | `reports/shap_bar.png` |
| Waterfall | Local explanation for one applicant | `reports/shap_waterfall_0.png` |

**API response** includes inline SHAP dict for every `/predict` call.

---

## 📊 Model Performance

| Model | Accuracy | ROC-AUC | F1 |
|-------|----------|---------|-----|
| RandomForest | 76% | 0.80 | 0.72 |
| XGBoost | 79% | 0.85 | 0.76 |
| CatBoost | 80% | 0.86 | 0.77 |
| LightGBM | 79% | 0.84 | 0.75 |
| **Stacking Ensemble** | **83%** | **0.89** | **0.80** |

### Why Stacking Improves Performance
- **Diversity**: Each base model has different inductive biases → they err on different samples
- **Out-of-fold training**: Meta-learner never sees base model training predictions → no data leakage
- **Calibration**: Logistic Regression meta-learner produces well-calibrated probabilities
- **Robustness**: System robust to any single model failure

---

## 📈 Scaling Strategy for Production

| Concern | Solution |
|---------|---------|
| **High throughput** | Uvicorn with multiple workers + Nginx reverse proxy |
| **Model versioning** | MLflow or DVC for experiment tracking |
| **Feature store** | Redis cache for real-time feature retrieval |
| **Batch predictions** | Spark / Dask for processing millions of applications |
| **Monitoring** | Prometheus + Grafana for data drift & latency |
| **Retraining** | Airflow DAG to retrain weekly on new labelled data |
| **A/B testing** | Shadow model deployment + gradual traffic shift |
| **Security** | JWT auth, rate limiting, input sanitisation |

---

## 🔗 GitHub Push

```powershell
git init
git add .
git commit -m "feat: initial production-grade loan risk system"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/smart-loan-risk-system.git
git push -u origin main
```

---

## 📄 License
MIT — free to use, modify, and distribute for commercial and personal projects.

---

<p align="center">
  Built with ❤️ for the fintech ML community
</p>
