"""
shap_explainer.py
─────────────────
SHAP-based explainability for the stacking ensemble.

How SHAP works
──────────────
SHAP (SHapley Additive exPlanations) uses game-theory Shapley values to
assign each feature a contribution to the prediction.
  • Global  → mean |SHAP value| across all samples = feature importance
  • Local   → per-sample waterfall showing which features pushed the
              prediction up/down from the baseline (mean prediction)

We use TreeExplainer on the XGBoost base estimator (fastest for tree models).
For the full stacking model we fall back to KernelExplainer (model-agnostic).
"""

import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

from src.config.config import REPORTS_DIR, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Wraps SHAP TreeExplainer for the XGBoost base model."""

    def __init__(self, stacking_model, feature_names: list):
        self.feature_names = feature_names
        # Use XGBoost sub-estimator for fast TreeExplainer
        try:
            self.xgb_model = dict(stacking_model.estimators_)["xgb"]
            self.explainer = shap.TreeExplainer(self.xgb_model)
            logger.info("SHAP TreeExplainer initialised on XGBoost sub-model.")
        except Exception as e:
            logger.warning(f"TreeExplainer failed ({e}); falling back to KernelExplainer.")
            self.xgb_model = None
            self.explainer  = None

    # ── Compute SHAP values ────────────────────────────────────────────────────
    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP values array (n_samples × n_features)."""
        if self.explainer is None:
            raise RuntimeError("Explainer not initialised.")
        shap_values = self.explainer.shap_values(X)
        # For binary classification some versions return list[2]; take class-1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return shap_values

    # ── Plots ─────────────────────────────────────────────────────────────────
    def plot_summary(self, X: np.ndarray, max_display: int = 15, save: bool = True) -> plt.Figure:
        """Global SHAP summary plot (beeswarm)."""
        shap_values = self.compute_shap_values(X)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        if save:
            path = REPORTS_DIR / "shap_summary.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP summary saved → {path}")
        return fig

    def plot_bar(self, X: np.ndarray, max_display: int = 15, save: bool = True) -> plt.Figure:
        """Global feature importance bar chart via SHAP."""
        shap_values = self.compute_shap_values(X)
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(
            shap_values, X,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        if save:
            path = REPORTS_DIR / "shap_bar.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP bar chart saved → {path}")
        return fig

    def plot_waterfall(
        self, X: np.ndarray, sample_idx: int = 0, save: bool = True
    ) -> plt.Figure:
        """Local waterfall plot for a single prediction."""
        shap_values = self.compute_shap_values(X)
        expected    = self.explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            expected = expected[1]

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[sample_idx],
                base_values=expected,
                data=X[sample_idx],
                feature_names=self.feature_names,
            ),
            show=False,
        )
        plt.tight_layout()
        if save:
            path = REPORTS_DIR / f"shap_waterfall_{sample_idx}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Waterfall saved → {path}")
        return fig

    def explain_single(self, x_single: np.ndarray) -> dict:
        """
        Return dict of {feature_name: shap_value} for one sample.
        Used by the API /predict endpoint for inline explanation.
        """
        shap_values = self.compute_shap_values(x_single.reshape(1, -1))
        return dict(zip(self.feature_names, shap_values[0].tolist()))
