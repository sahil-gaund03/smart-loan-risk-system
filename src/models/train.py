"""
train.py
────────
End-to-end training pipeline:
  1. Load raw data
  2. Feature engineering
  3. Preprocessing + SMOTE
  4. Optuna hyperparameter tuning (XGBoost as representative model)
  5. Build stacking ensemble with tuned params
  6. Train + save model
"""

import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import optuna
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.config.config import (
    RANDOM_STATE, CV_FOLDS, OPTUNA_TRIALS, MODELS_DIR, LOG_FORMAT, LOG_LEVEL
)
from src.data.data_loader import load_raw_data
from src.data.preprocess import LoanPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.stacking_model import build_stacking_model, save_model
from src.models.evaluate import evaluate_model

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Optuna Objective ──────────────────────────────────────────────────────────
def _xgb_objective(trial, X_train, y_train, cv_folds):
    """Optimise XGBoost ROC-AUC via Optuna."""
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 100, 400),
        "max_depth":       trial.suggest_int("max_depth", 3, 8),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":trial.suggest_int("min_child_weight", 1, 10),
        "gamma":           trial.suggest_float("gamma", 0, 5),
        "random_state":    RANDOM_STATE,
        "use_label_encoder": False,
        "eval_metric":     "logloss",
        "n_jobs":          -1,
    }
    model = XGBClassifier(**params)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def tune_xgboost(X_train, y_train, n_trials: int = OPTUNA_TRIALS) -> dict:
    """Run Optuna study and return best params for XGBoost."""
    logger.info(f"Starting Optuna tuning ({n_trials} trials) …")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(
        lambda trial: _xgb_objective(trial, X_train, y_train, CV_FOLDS),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    logger.info(f"Best XGBoost ROC-AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return study.best_params


# ── Main Training Pipeline ────────────────────────────────────────────────────
def train(n_optuna_trials: int = OPTUNA_TRIALS):
    """Full training pipeline. Returns trained model and metrics dict."""

    # 1. Load
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading data")
    df = load_raw_data()

    # 2. Feature engineering
    logger.info("STEP 2 — Feature engineering")
    fe = FeatureEngineer()
    df = fe.transform(df)

    # 3. Preprocessing + SMOTE
    logger.info("STEP 3 — Preprocessing + SMOTE")
    from sklearn.model_selection import train_test_split
    from src.config.config import TARGET_COLUMN, TEST_SIZE

    preprocessor = LoanPreprocessor()
    X_all, y_all = preprocessor.fit_transform(df, apply_smote=False)

    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
    )

    # SMOTE only on training split
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)
    logger.info(f"Train size after SMOTE: {X_train.shape}, Test size: {X_test.shape}")

    # 4. Optuna tuning for XGBoost
    logger.info("STEP 4 — Optuna hyperparameter tuning (XGBoost)")
    best_xgb_params = tune_xgboost(X_train, y_train, n_trials=n_optuna_trials)

    # 5. Build stacking model
    logger.info("STEP 5 — Building stacking ensemble")
    model = build_stacking_model(xgb_params=best_xgb_params)

    # 6. Train
    logger.info("STEP 6 — Training stacking ensemble …")
    model.fit(X_train, y_train)

    # 7. Evaluate
    logger.info("STEP 7 — Evaluation")
    metrics = evaluate_model(model, X_test, y_test)

    # 8. Save
    logger.info("STEP 8 — Saving artifacts")
    preprocessor.save()
    save_model(model)

    # Also save test set for notebook usage
    joblib.dump((X_test, y_test), MODELS_DIR / "test_set.joblib")

    logger.info("=" * 60)
    logger.info("Training complete!")
    return model, metrics


if __name__ == "__main__":
    model, metrics = train(n_optuna_trials=OPTUNA_TRIALS)
    print("\nFinal metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
