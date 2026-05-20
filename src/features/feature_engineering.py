"""
feature_engineering.py
──────────────────────
Domain-driven features for loan risk assessment.

New features:
  • income_to_loan_ratio     – higher = safer
  • debt_burden_score        – higher = riskier
  • credit_utilization       – % of credit window used
  • financial_stability_score – composite safety metric
  • fraud_risk_score         – rule-based fraud indicator (0-100)
"""

import logging
import numpy as np
import pandas as pd

from src.config.config import LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ── Ordinal mappings for categorical columns ───────────────────────────────────
SAVING_MAP = {
    "little": 1, "moderate": 2, "quite rich": 3, "rich": 4,
}
CHECKING_MAP = {
    "little": 1, "moderate": 2, "rich": 3,
}
HOUSING_MAP   = {"free": 0, "rent": 1, "own": 2}
SEX_MAP       = {"female": 0, "male": 1}


class FeatureEngineer:
    """
    Adds domain-driven engineered features to a loan DataFrame.
    Designed to run BEFORE encoding/scaling so it can read raw column values.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with added engineered columns."""
        df = df.copy()
        df = self._map_ordinals(df)
        df = self._income_to_loan_ratio(df)
        df = self._debt_burden_score(df)
        df = self._credit_utilization(df)
        df = self._financial_stability_score(df)
        df = self._fraud_risk_score(df)
        logger.info(f"Engineered features added. New shape: {df.shape}")
        return df

    # ── Individual feature builders ───────────────────────────────────────────
    @staticmethod
    def _map_ordinals(df: pd.DataFrame) -> pd.DataFrame:
        """Convert text categoricals to ordered numerics for computation."""
        df["saving_num"]   = df["Saving accounts"].map(SAVING_MAP).fillna(0).astype(float)
        df["checking_num"] = df["Checking account"].map(CHECKING_MAP).fillna(0).astype(float)
        df["housing_num"]  = df["Housing"].map(HOUSING_MAP).fillna(1).astype(float)
        df["sex_num"]      = df["Sex"].map(SEX_MAP).fillna(0).astype(float)
        return df

    @staticmethod
    def _income_to_loan_ratio(df: pd.DataFrame) -> pd.DataFrame:
        """
        Proxy for income = Job * 1000 (higher skilled job → higher income proxy).
        Ratio = income_proxy / credit_amount.  High ratio → safer.
        """
        income_proxy = (df["Job"].fillna(1) + 1) * 1_000
        df["income_to_loan_ratio"] = income_proxy / df["Credit amount"].replace(0, np.nan)
        df["income_to_loan_ratio"] = df["income_to_loan_ratio"].fillna(0).clip(0, 10)
        return df

    @staticmethod
    def _debt_burden_score(df: pd.DataFrame) -> pd.DataFrame:
        """
        Debt burden = Duration (months) × Credit amount / 10_000.
        Higher → heavier debt burden → riskier.
        """
        df["debt_burden_score"] = (df["Duration"] * df["Credit amount"]) / 10_000
        df["debt_burden_score"] = df["debt_burden_score"].clip(0, 500)
        return df

    @staticmethod
    def _credit_utilization(df: pd.DataFrame) -> pd.DataFrame:
        """
        Credit utilization = Credit amount / (Age × 500).
        Older applicants relative to loan = lower utilization.
        """
        df["credit_utilization"] = df["Credit amount"] / (df["Age"].replace(0, 1) * 500)
        df["credit_utilization"] = df["credit_utilization"].clip(0, 5)
        return df

    @staticmethod
    def _financial_stability_score(df: pd.DataFrame) -> pd.DataFrame:
        """
        Composite stability:  saving + checking + housing + (job * 0.5).
        Range roughly 0-8.  Higher = more stable.
        """
        df["financial_stability_score"] = (
            df["saving_num"]
            + df["checking_num"]
            + df["housing_num"]
            + df["Job"].fillna(1) * 0.5
        )
        return df

    @staticmethod
    def _fraud_risk_score(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule-based fraud risk score (0-100).
        Rules:
          +30  Credit amount > 90th percentile
          +25  Duration > 48 months AND no checking account
          +20  saving_num == 0 AND checking_num == 0
          +15  Age < 22
          +10  Job == 0 (unskilled non-resident)
        """
        p90 = df["Credit amount"].quantile(0.90)
        score = pd.Series(0, index=df.index, dtype=float)
        score += (df["Credit amount"] > p90) * 30
        score += ((df["Duration"] > 48) & (df["checking_num"] == 0)) * 25
        score += ((df["saving_num"] == 0) & (df["checking_num"] == 0)) * 20
        score += (df["Age"] < 22) * 15
        score += (df["Job"] == 0) * 10
        df["fraud_risk_score"] = score.clip(0, 100)
        return df
