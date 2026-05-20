"""
main.py  —  Smart Loan Risk Prediction API
──────────────────────────────────────────
Endpoints:
  GET  /health         – liveness probe
  GET  /model-info     – model metadata
  POST /predict        – loan risk prediction + SHAP explanation
  POST /fraud-check    – standalone fraud detection

Run with:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from src.config.config import (
    API_TITLE, API_VERSION,
    STACKING_MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH,
    HIGH_RISK_THRESHOLD, LOG_FORMAT, LOG_LEVEL,
)
from src.features.feature_engineering import FeatureEngineer
from src.fraud.fraud_detector import FraudDetector

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ── Global model registry (loaded once at startup) ────────────────────────────
_registry: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup; clean up on shutdown."""
    logger.info("Loading model artifacts …")
    try:
        _registry["model"]        = joblib.load(STACKING_MODEL_PATH)
        _registry["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
        _registry["features"]     = joblib.load(FEATURE_NAMES_PATH)
        logger.info("All artifacts loaded successfully.")
    except FileNotFoundError as exc:
        logger.error(f"Artifact not found: {exc}. Run training first.")
    yield
    _registry.clear()
    logger.info("Model registry cleared on shutdown.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Production-grade loan risk & fraud detection API with SHAP explainability.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ApplicantInput(BaseModel):
    Age: int              = Field(..., ge=18, le=100,  example=35)
    Sex: str              = Field(...,                  example="male")
    Job: int              = Field(..., ge=0, le=3,      example=2)
    Housing: str          = Field(...,                  example="own")
    Saving_accounts: Optional[str]  = Field(None,       example="little")
    Checking_account: Optional[str] = Field(None,       example="moderate")
    Credit_amount: float  = Field(..., gt=0,            example=5000)
    Duration: int         = Field(..., gt=0,            example=24)
    Purpose: str          = Field(...,                  example="car")

    class Config:
        schema_extra = {
            "example": {
                "Age": 35, "Sex": "male", "Job": 2,
                "Housing": "own", "Saving_accounts": "moderate",
                "Checking_account": "little",
                "Credit_amount": 5000, "Duration": 24, "Purpose": "car",
            }
        }


class PredictionResponse(BaseModel):
    risk_label:      str
    risk_probability: float
    risk_score_pct:  float
    shap_explanation: dict
    fraud_result:    dict
    model_version:   str


class FraudCheckResponse(BaseModel):
    fraud_score:   float
    is_fraud_flag: bool
    flags:         list
    recommendation: str


class HealthResponse(BaseModel):
    status:          str
    model_loaded:    bool
    api_version:     str


class ModelInfoResponse(BaseModel):
    model_type:  str
    base_models: list
    features:    list
    threshold:   float


# ── Helpers ───────────────────────────────────────────────────────────────────
def _applicant_to_df(inp: ApplicantInput):
    import pandas as pd
    raw = {
        "Age":              inp.Age,
        "Sex":              inp.Sex,
        "Job":              inp.Job,
        "Housing":          inp.Housing,
        "Saving accounts":  inp.Saving_accounts,
        "Checking account": inp.Checking_account,
        "Credit amount":    inp.Credit_amount,
        "Duration":         inp.Duration,
        "Purpose":          inp.Purpose,
    }
    return pd.DataFrame([raw])


def _require_model():
    if "model" not in _registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please run training first.",
        )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok",
        model_loaded="model" in _registry,
        api_version=API_VERSION,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    _require_model()
    return ModelInfoResponse(
        model_type="StackingClassifier",
        base_models=["RandomForest", "XGBoost", "CatBoost", "LightGBM"],
        features=_registry.get("features", []),
        threshold=HIGH_RISK_THRESHOLD,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(inp: ApplicantInput):
    _require_model()
    try:
        df          = _applicant_to_df(inp)
        fe          = FeatureEngineer()
        df_eng      = fe.transform(df)

        preprocessor = _registry["preprocessor"]
        X            = preprocessor.transform(df_eng)

        model        = _registry["model"]
        proba        = model.predict_proba(X)[0][1]
        label        = "HIGH RISK" if proba >= HIGH_RISK_THRESHOLD else "LOW RISK"

        # SHAP (fast path via XGBoost sub-model)
        shap_dict: dict = {}
        try:
            import shap
            xgb_est = dict(model.estimators_).get("xgb")
            if xgb_est:
                explainer  = shap.TreeExplainer(xgb_est)
                sv         = explainer.shap_values(X)
                if isinstance(sv, list):
                    sv = sv[1]
                feats = _registry.get("features", [f"f{i}" for i in range(X.shape[1])])
                shap_dict = {feats[i]: round(float(sv[0][i]), 4) for i in range(len(feats))}
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")
            shap_dict = {}

        # Fraud check
        applicant_dict = df_eng.iloc[0].to_dict()
        applicant_dict["income_to_loan_ratio"] = float(df_eng.iloc[0].get("income_to_loan_ratio", 1))
        fraud_result = FraudDetector().detect(applicant_dict)

        return PredictionResponse(
            risk_label=label,
            risk_probability=round(float(proba), 4),
            risk_score_pct=round(float(proba) * 100, 2),
            shap_explanation=shap_dict,
            fraud_result={
                "fraud_score":    fraud_result.fraud_score,
                "is_fraud_flag":  fraud_result.is_fraud_flag,
                "flags":          fraud_result.flags,
                "recommendation": fraud_result.recommendation,
            },
            model_version=API_VERSION,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fraud-check", response_model=FraudCheckResponse, tags=["Fraud"])
async def fraud_check(inp: ApplicantInput):
    try:
        df        = _applicant_to_df(inp)
        fe        = FeatureEngineer()
        df_eng    = fe.transform(df)
        applicant = df_eng.iloc[0].to_dict()
        result    = FraudDetector().detect(applicant)
        return FraudCheckResponse(
            fraud_score=result.fraud_score,
            is_fraud_flag=result.is_fraud_flag,
            flags=result.flags,
            recommendation=result.recommendation,
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs."},
    )
