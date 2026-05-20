"""
02_feature_engineering.py
Run: python notebooks/02_feature_engineering.py  (from project root)
"""
# %% Imports
import sys; sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings('ignore')

from src.data.data_loader import load_raw_data
from src.features.feature_engineering import FeatureEngineer
from src.config.config import REPORTS_DIR, PROC_DATA_PATH

sns.set_theme(style='whitegrid')

# %% Load and engineer
df  = load_raw_data()
fe  = FeatureEngineer()
dfe = fe.transform(df)

print("Engineered features added:")
eng_cols = ['income_to_loan_ratio','debt_burden_score','credit_utilization',
            'financial_stability_score','fraud_risk_score']
print(dfe[eng_cols].describe())

# %% Save processed data
dfe.to_csv(PROC_DATA_PATH, index=False)
print(f"\nProcessed data saved → {PROC_DATA_PATH}")

# %% Feature distributions
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Engineered Features Distribution', fontsize=15, fontweight='bold')

for i, col in enumerate(eng_cols):
    ax = axes[i // 3][i % 3]
    for risk, color in [(0,'#16a34a'),(1,'#dc2626')]:
        subset = dfe[dfe['loan_status'] == risk][col]
        ax.hist(subset, bins=20, alpha=0.6, color=color,
                label=['Low Risk','High Risk'][risk])
    ax.set_title(col.replace('_',' ').title())
    ax.legend(fontsize=8)

# Hide unused subplot
axes[1][2].axis('off')
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'engineered_features.png', dpi=150, bbox_inches='tight')
print(f"Saved → {REPORTS_DIR / 'engineered_features.png'}")

# %% Fraud score distribution
fig2, ax = plt.subplots(figsize=(8, 5))
ax.hist(dfe[dfe['loan_status']==0]['fraud_risk_score'], bins=20,
        alpha=0.6, color='#16a34a', label='Low Risk')
ax.hist(dfe[dfe['loan_status']==1]['fraud_risk_score'], bins=20,
        alpha=0.6, color='#dc2626', label='High Risk')
ax.axvline(60, color='black', linestyle='--', label='Fraud Threshold (60)')
ax.set_title('Fraud Risk Score Distribution by Loan Status')
ax.set_xlabel('Fraud Risk Score')
ax.legend()
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'fraud_score_dist.png', dpi=150)
print(f"Saved → {REPORTS_DIR / 'fraud_score_dist.png'}")

print("\n✅ Feature engineering notebook complete!")
