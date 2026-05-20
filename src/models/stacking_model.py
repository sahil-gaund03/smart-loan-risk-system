"""
stacking_model.py
─────────────────
Stacking Ensemble:
  Base learners: RandomForest, XGBoost, CatBoost, LightGBM
  Meta learner:  Logistic Regression

Why stacking improves performance
──────────────────────────────────
Each base model captures different patterns:
  • RandomForest – reduces variance via bagging, good on noisy data
  • XGBoost      – gradient boosting; strong on tabular, handles missing values
  • CatBoost     – native categorical handling, robust out-of-the-box
  • LightGBM     – fastest boosting, leaf-wise growth, great for large data

The meta-learner (Logistic Regression) learns *when to trust* each model:
it sees out-of-fold predictions (not seen during base training) so it
learns the confidence profile of each model without overfitting.
Result: lower bias AND lower variance than any single model.
"""

import logging
import joblib
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from src.config.config import RANDOM_STATE, STACKING_MODEL_PATH, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def build_stacking_model(
    rf_params:  dict | None = None,
    xgb_params: dict | None = None,
    cat_params: dict | None = None,
    lgb_params: dict | None = None,
) -> StackingClassifier:
    """
    Build a StackingClassifier with tunable base estimator params.
    Default params are production-ready starting points.
    """
    rf_defaults = dict(
        n_estimators=200, max_depth=8, min_samples_split=4,
        random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
    )
    xgb_defaults = dict(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
        use_label_encoder=False, eval_metric="logloss",
    )
    cat_defaults = dict(
        iterations=200, depth=5, learning_rate=0.05,
        random_seed=RANDOM_STATE, verbose=0,
    )
    lgb_defaults = dict(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
        verbose=-1,
    )

    rf_p  = {**rf_defaults,  **(rf_params  or {})}
    xgb_p = {**xgb_defaults, **(xgb_params or {})}
    cat_p = {**cat_defaults,  **(cat_params or {})}
    lgb_p = {**lgb_defaults,  **(lgb_params or {})}

    estimators = [
        ("rf",  RandomForestClassifier(**rf_p)),
        ("xgb", XGBClassifier(**xgb_p)),
        ("cat", CatBoostClassifier(**cat_p)),
        ("lgb", LGBMClassifier(**lgb_p)),
    ]

    meta_learner = LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
    )

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )
    logger.info("Stacking model built with 4 base learners + Logistic Regression meta.")
    return stacking


def save_model(model: StackingClassifier, path: Path = STACKING_MODEL_PATH):
    joblib.dump(model, path)
    logger.info(f"Model saved → {path}")


def load_model(path: Path = STACKING_MODEL_PATH) -> StackingClassifier:
    model = joblib.load(path)
    logger.info(f"Model loaded ← {path}")
    return model
