"""
data_loader.py
──────────────
Loads raw German Credit dataset and applies initial type fixes.
The German Credit dataset has no explicit 'loan_status' column; we derive it
from the 'Purpose' & account information as a proxy target variable so the
full ML pipeline can be demonstrated end-to-end.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

from src.config.config import RAW_DATA_PATH, TARGET_COLUMN, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and does minimal cleaning on the raw loan dataset."""

    def __init__(self, filepath: Path = RAW_DATA_PATH):
        self.filepath = filepath

    # ── Public API ────────────────────────────────────────────────────────────
    def load(self) -> pd.DataFrame:
        """Return a cleaned DataFrame ready for preprocessing."""
        logger.info(f"Loading data from {self.filepath}")
        df = pd.read_csv(self.filepath, index_col=0)
        logger.info(f"Raw shape: {df.shape}")
        df = self._rename_columns(df)
        df = self._create_target(df)
        df = self._fix_dtypes(df)
        logger.info(f"Loaded shape after cleaning: {df.shape}")
        return df

    # ── Private Helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardise column names (strip whitespace)."""
        df.columns = [c.strip() for c in df.columns]
        return df

    @staticmethod
    def _create_target(df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive a binary loan_status target.
        High-risk (1) proxy logic based on:
          - Duration > 24 months  AND Credit amount > median
          - OR Checking account is NA (no account → risky)
          - OR Saving accounts is NA
        This is a realistic demo proxy for a dataset that
        originally has a separate risk column in other versions.
        """
        median_credit = df["Credit amount"].median()

        high_risk_mask = (
            ((df["Duration"] > 24) & (df["Credit amount"] > median_credit))
            | (df["Checking account"].isna())
        )
        df[TARGET_COLUMN] = high_risk_mask.astype(int)
        logger.info(
            f"Target distribution:\n{df[TARGET_COLUMN].value_counts(normalize=True).round(3)}"
        )
        return df

    @staticmethod
    def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Cast columns to appropriate dtypes."""
        int_cols = ["Age", "Credit amount", "Duration", "Job"]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df


# ── Convenience function ──────────────────────────────────────────────────────
def load_raw_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return DataLoader(filepath).load()


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
    print(df.dtypes)
    print(df[TARGET_COLUMN].value_counts())
