"""
evaluate.py
───────────
Model evaluation utilities: metrics, confusion matrix, ROC-AUC, plots.
"""

import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, ConfusionMatrixDisplay,
)

from src.config.config import REPORTS_DIR, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute and log all classification metrics. Return as dict."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }

    logger.info("\n" + "=" * 50)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("=" * 50)
    for k, v in metrics.items():
        logger.info(f"  {k.upper():12s}: {v:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred,
                                             target_names=["Low Risk", "High Risk"]))
    return metrics


def plot_confusion_matrix(model, X_test, y_test, save: bool = True) -> plt.Figure:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Low Risk", "High Risk"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix – Stacking Ensemble", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        path = REPORTS_DIR / "confusion_matrix.png"
        fig.savefig(path, dpi=150)
        logger.info(f"Saved confusion matrix → {path}")
    return fig


def plot_roc_curve(model, X_test, y_test, save: bool = True) -> plt.Figure:
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve – Stacking Ensemble")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save:
        path = REPORTS_DIR / "roc_curve.png"
        fig.savefig(path, dpi=150)
        logger.info(f"Saved ROC curve → {path}")
    return fig


def plot_feature_importance(model, feature_names: list, save: bool = True) -> plt.Figure:
    """
    Extract feature importances from the XGBoost base estimator
    inside the stacking model.
    """
    try:
        xgb_est = dict(model.estimators_).get("xgb", None)
        if xgb_est is None:
            logger.warning("XGBoost estimator not found in stacking model.")
            return None
        importances = xgb_est.feature_importances_
        n = min(len(feature_names), len(importances))
        idx = np.argsort(importances[:n])[-15:]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(
            [feature_names[i] for i in idx],
            importances[idx],
            color="#2563EB",
        )
        ax.set(title="Feature Importances (XGBoost)", xlabel="Importance")
        plt.tight_layout()
        if save:
            path = REPORTS_DIR / "feature_importance.png"
            fig.savefig(path, dpi=150)
            logger.info(f"Saved feature importance → {path}")
        return fig
    except Exception as e:
        logger.warning(f"Could not plot feature importance: {e}")
        return None
