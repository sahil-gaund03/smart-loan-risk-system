"""
preprocess.py
─────────────
Full preprocessing pipeline:
  1. Missing-value imputation
  2. Outlier capping (IQR)
  3. Encoding (OrdinalEncoder for tree models)
  4. Standard scaling for numeric features
  5. SMOTE oversampling for class imbalance
"""

import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    PREPROCESSOR_PATH,
    FEATURE_NAMES_PATH,
    RANDOM_STATE,
    LOG_FORMAT,
    LOG_LEVEL,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ── Outlier Capper ────────────────────────────────────────────────────────────
class IQRCapper:
    """Caps numeric outliers at [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.bounds_: dict = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in X.select_dtypes(include=np.number).columns:
            q1, q3 = X[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.bounds_[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lo, hi)
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X).transform(X)


# ── Preprocessor Builder ──────────────────────────────────────────────────────
def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Returns a fitted-ready ColumnTransformer.
    Numeric  → impute (median) → scale
    Categoric → impute (most_frequent) → ordinal encode
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe,  numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


# ── Main Preprocessing Class ──────────────────────────────────────────────────
class LoanPreprocessor:
    """End-to-end preprocessing: cap → transform → SMOTE."""

    def __init__(
        self,
        numeric_cols: list = NUMERIC_FEATURES,
        categorical_cols: list = CATEGORICAL_FEATURES,
    ):
        self.numeric_cols    = numeric_cols
        self.categorical_cols = categorical_cols
        self.capper_         = IQRCapper()
        self.preprocessor_   = build_preprocessor(numeric_cols, categorical_cols)
        self.feature_names_: list = []

    # ── fit_transform ─────────────────────────────────────────────────────────
    def fit_transform(
        self, df: pd.DataFrame, apply_smote: bool = True
    ):
        """
        Fit on training data and return (X_resampled, y_resampled).
        """
        logger.info("Fitting preprocessor …")
        y = df[TARGET_COLUMN].values

        # 1. Cap outliers on numeric cols only
        numeric_present = [c for c in self.numeric_cols if c in df.columns]
        df[numeric_present] = self.capper_.fit_transform(df[numeric_present])

        # 2. Build feature matrix
        cols_needed = [c for c in self.numeric_cols + self.categorical_cols if c in df.columns]
        X_raw = df[cols_needed]

        # 3. Column transform
        X = self.preprocessor_.fit_transform(X_raw)

        # 4. Feature names
        self._set_feature_names(cols_needed)

        # 5. SMOTE
        if apply_smote:
            logger.info("Applying SMOTE to handle class imbalance …")
            smote = SMOTE(random_state=RANDOM_STATE)
            X, y = smote.fit_resample(X, y)
            logger.info(f"After SMOTE: {np.bincount(y)}")

        return X, y

    # ── transform ─────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new (unseen) data — no fitting, no SMOTE."""
        numeric_present = [c for c in self.numeric_cols if c in df.columns]
        df = df.copy()
        df[numeric_present] = self.capper_.transform(df[numeric_present])
        cols_needed = [c for c in self.numeric_cols + self.categorical_cols if c in df.columns]
        return self.preprocessor_.transform(df[cols_needed])

    # ── persist ───────────────────────────────────────────────────────────────
    def save(self):
        joblib.dump(self, PREPROCESSOR_PATH)
        joblib.dump(self.feature_names_, FEATURE_NAMES_PATH)
        logger.info(f"Preprocessor saved → {PREPROCESSOR_PATH}")

    @classmethod
    def load(cls) -> "LoanPreprocessor":
        obj = joblib.load(PREPROCESSOR_PATH)
        logger.info(f"Preprocessor loaded ← {PREPROCESSOR_PATH}")
        return obj

    # ── helpers ───────────────────────────────────────────────────────────────
    def _set_feature_names(self, cols_used: list):
        num_present  = [c for c in self.numeric_cols if c in cols_used]
        cat_present  = [c for c in self.categorical_cols if c in cols_used]
        self.feature_names_ = num_present + cat_present
