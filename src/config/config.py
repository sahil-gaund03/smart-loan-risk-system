"""
Central configuration for Smart Loan Risk Prediction System.
All paths, hyperparameters, and constants live here.
"""

import os
from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]   # smart-loan-risk-system/

# ── Data Paths ────────────────────────────────────────────────────────────────
DATA_DIR       = BASE_DIR / "data"
RAW_DATA_DIR   = DATA_DIR / "raw"
PROC_DATA_DIR  = DATA_DIR / "processed"

RAW_DATA_PATH  = RAW_DATA_DIR / "german_credit_data.csv"
PROC_DATA_PATH = PROC_DATA_DIR / "processed_loan_data.csv"

# ── Model Paths ───────────────────────────────────────────────────────────────
MODELS_DIR     = BASE_DIR / "models"
STACKING_MODEL_PATH   = MODELS_DIR / "stacking_model.joblib"
PREPROCESSOR_PATH     = MODELS_DIR / "preprocessor.joblib"
FEATURE_NAMES_PATH    = MODELS_DIR / "feature_names.joblib"

# ── Reports ───────────────────────────────────────────────────────────────────
REPORTS_DIR    = BASE_DIR / "reports"

# ── Ensure directories exist ──────────────────────────────────────────────────
for d in [RAW_DATA_DIR, PROC_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Target & Feature Columns ──────────────────────────────────────────────────
TARGET_COLUMN = "loan_status"          # 1 = High-Risk, 0 = Low-Risk

NUMERIC_FEATURES = [
    "Age",
    "Credit amount",
    "Duration",
    "Job",
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Purpose",
]

ENGINEERED_FEATURES = [
    "income_to_loan_ratio",
    "debt_burden_score",
    "credit_utilization",
    "financial_stability_score",
    "fraud_risk_score",
]

# ── Model Training ────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
CV_FOLDS       = 5
OPTUNA_TRIALS  = 30          # reduce for speed; increase for better tuning

# ── Risk Thresholds ───────────────────────────────────────────────────────────
HIGH_RISK_THRESHOLD  = 0.50  # probability ≥ this → HIGH RISK
FRAUD_SCORE_THRESHOLD = 60   # fraud score ≥ this → FLAG

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_TITLE   = "Smart Loan Risk Prediction API"
API_VERSION = "1.0.0"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
