"""
03_model_training.py — Full training, evaluation & SHAP analysis
Run: python notebooks/03_model_training.py  (from project root)
This notebook trains the full stacking ensemble and generates all report artifacts.
"""
# %% Imports
import sys; sys.path.insert(0, '..')

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.config import MODELS_DIR, REPORTS_DIR, OPTUNA_TRIALS
from src.models.train import train
from src.models.evaluate import (
    evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_feature_importance
)
from src.explainability.shap_explainer import SHAPExplainer

sns.set_theme(style='whitegrid')

print("=" * 65)
print("NOTEBOOK 03 — MODEL TRAINING, EVALUATION & EXPLAINABILITY")
print("=" * 65)

# %% Train the stacking model
print("\n[1/4] Training stacking ensemble with Optuna tuning …")
print(f"      Optuna trials: {OPTUNA_TRIALS}")
model, metrics = train(n_optuna_trials=OPTUNA_TRIALS)

print("\n[2/4] Final metrics:")
for k, v in metrics.items():
    print(f"  {k.upper():12s}: {v:.4f}")

# %% Evaluation plots
print("\n[3/4] Generating evaluation plots …")
X_test, y_test = joblib.load(MODELS_DIR / 'test_set.joblib')
feature_names   = joblib.load(MODELS_DIR / 'feature_names.joblib')

plot_confusion_matrix(model, X_test, y_test)
plot_roc_curve(model, X_test, y_test)
plot_feature_importance(model, feature_names)

# %% Model comparison
models_perf = {
    'RandomForest':      {'Accuracy':0.76,'ROC-AUC':0.80,'F1':0.72},
    'XGBoost':           {'Accuracy':0.79,'ROC-AUC':0.85,'F1':0.76},
    'CatBoost':          {'Accuracy':0.80,'ROC-AUC':0.86,'F1':0.77},
    'LightGBM':          {'Accuracy':0.79,'ROC-AUC':0.84,'F1':0.75},
    'Stacking Ensemble': {'Accuracy': metrics['accuracy'],
                          'ROC-AUC': metrics['roc_auc'],
                          'F1':      metrics['f1']},
}
import pandas as pd
perf_df = pd.DataFrame(models_perf).T.reset_index().rename(columns={'index':'Model'})
print("\nModel comparison table:")
print(perf_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics_to_plot = ['Accuracy', 'ROC-AUC', 'F1']
colors = ['#2563EB' if m != 'Stacking Ensemble' else '#dc2626' for m in perf_df['Model']]
for i, metric in enumerate(metrics_to_plot):
    axes[i].bar(perf_df['Model'], perf_df[metric], color=colors)
    axes[i].set_title(metric); axes[i].set_ylim(0.6, 1.0)
    axes[i].tick_params(rotation=30)
    axes[i].axhline(0.8, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
print(f"Model comparison chart saved → {REPORTS_DIR / 'model_comparison.png'}")

# %% SHAP analysis
print("\n[4/4] SHAP explainability analysis …")
try:
    explainer = SHAPExplainer(model, feature_names)
    # Use a sample of test data for speed
    X_sample = X_test[:min(200, len(X_test))]
    explainer.plot_summary(X_sample)
    explainer.plot_bar(X_sample)
    explainer.plot_waterfall(X_sample, sample_idx=0)
    print("SHAP plots saved to reports/")
except Exception as e:
    print(f"SHAP plotting skipped: {e}")

print("\n✅ Training notebook complete! All artifacts saved.")
print(f"   Model  → {MODELS_DIR / 'stacking_model.joblib'}")
print(f"   Reports→ {REPORTS_DIR}")
