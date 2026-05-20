"""
01_eda.py  —  Exploratory Data Analysis
Run: python notebooks/01_eda.py  (from project root)
Or open as Jupyter notebook after converting with jupytext.

# %% Imports
import sys; sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings('ignore')

from src.data.data_loader import load_raw_data
from src.config.config import REPORTS_DIR

sns.set_theme(style='whitegrid', palette='muted')

# %% Load data
df = load_raw_data()
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:\n", df.head())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDescriptive stats:\n", df.describe())

# %% Target distribution
print("\nTarget distribution:\n", df['loan_status'].value_counts())
print("\nTarget % :\n", df['loan_status'].value_counts(normalize=True).round(3))

# %% EDA overview plots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('EDA — German Credit Dataset', fontsize=16, fontweight='bold')

axes[0,0].hist(df['Age'].dropna(), bins=20, color='#2563EB', edgecolor='white')
axes[0,0].set_title('Age Distribution'); axes[0,0].set_xlabel('Age')

axes[0,1].hist(df['Credit amount'].dropna(), bins=25, color='#16a34a', edgecolor='white')
axes[0,1].set_title('Credit Amount Distribution'); axes[0,1].set_xlabel('€')

axes[0,2].hist(df['Duration'].dropna(), bins=15, color='#f59e0b', edgecolor='white')
axes[0,2].set_title('Loan Duration'); axes[0,2].set_xlabel('months')

counts = df['loan_status'].value_counts()
axes[1,0].bar(['Low Risk','High Risk'], counts.values, color=['#16a34a','#dc2626'])
axes[1,0].set_title('Target Variable')

df['Saving accounts'].fillna('NA').value_counts().plot(kind='bar', ax=axes[1,1], color='#8b5cf6')
axes[1,1].set_title('Saving Accounts'); axes[1,1].tick_params(rotation=30)

df['Purpose'].value_counts().plot(kind='bar', ax=axes[1,2], color='#0891b2')
axes[1,2].set_title('Loan Purpose'); axes[1,2].tick_params(rotation=40)

plt.tight_layout()
plt.savefig(REPORTS_DIR / 'eda_overview.png', dpi=150, bbox_inches='tight')
print(f"Saved → {REPORTS_DIR / 'eda_overview.png'}")

# %% Correlation heatmap
num_cols = ['Age', 'Credit amount', 'Duration', 'Job', 'loan_status']
corr = df[num_cols].corr()
fig2, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, center=0)
ax.set_title('Correlation Matrix')
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'correlation_heatmap.png', dpi=150)
print(f"Saved → {REPORTS_DIR / 'correlation_heatmap.png'}")

# %% Credit amount by risk
fig3, ax3 = plt.subplots(figsize=(8, 5))
for risk, color in [(0,'#16a34a'),(1,'#dc2626')]:
    data = df[df['loan_status']==risk]['Credit amount']
    ax3.hist(data, bins=25, alpha=0.6, color=color, label=['Low Risk','High Risk'][risk])
ax3.set_title('Credit Amount by Risk Class')
ax3.set_xlabel('Credit Amount (€)')
ax3.legend()
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'credit_by_risk.png', dpi=150)
print(f"Saved → {REPORTS_DIR / 'credit_by_risk.png'}")

print("\n✅ EDA notebook complete!")
