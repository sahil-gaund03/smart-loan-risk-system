"""
fraud_detector.py
─────────────────
Rule-based fraud detection layer.
Returns a fraud risk score (0-100) and individual flags.

Rules (additive weights):
  +30  Loan amount unusually high (> 90th percentile of training data)
  +25  Checking account missing + Duration > 48 months
  +20  Both saving & checking accounts missing
  +15  Applicant age < 22
  +10  Unskilled / no residence (Job == 0)
  +15  Suspiciously low income-to-loan ratio (< 0.2)
  +20  Fraud risk score from feature engineering already flagged high
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.config.config import FRAUD_SCORE_THRESHOLD, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class FraudResult:
    fraud_score: float                  # 0-100
    is_fraud_flag: bool
    flags: list[str] = field(default_factory=list)
    recommendation: str = ""


class FraudDetector:
    """
    Evaluates an applicant record against rule-based fraud indicators.
    Accepts a dict with raw/engineered feature values.
    """

    # Training-set 90th percentile for credit amount (German dataset ~7660)
    CREDIT_P90 = 7_660

    def detect(self, applicant: dict) -> FraudResult:
        """
        Parameters
        ----------
        applicant : dict
            Keys should match model features (raw or engineered).

        Returns
        -------
        FraudResult
        """
        score  = 0.0
        flags  = []

        credit_amount  = float(applicant.get("Credit amount", 0))
        duration       = float(applicant.get("Duration", 0))
        age            = float(applicant.get("Age", 30))
        job            = float(applicant.get("Job", 1))
        saving         = str(applicant.get("Saving accounts", "")).lower()
        checking       = str(applicant.get("Checking account", "")).lower()
        i2l            = float(applicant.get("income_to_loan_ratio", 1.0))
        fe_fraud_score = float(applicant.get("fraud_risk_score", 0))

        # Rule 1: High loan amount
        if credit_amount > self.CREDIT_P90:
            score += 30
            flags.append(f"Loan amount ({credit_amount:,.0f}) exceeds 90th percentile threshold")

        # Rule 2: Long duration + no checking account
        if duration > 48 and (checking in ("", "na", "nan", "none")):
            score += 25
            flags.append("Long loan duration with no checking account — high default risk")

        # Rule 3: No savings AND no checking
        if (saving in ("", "na", "nan", "none")) and (checking in ("", "na", "nan", "none")):
            score += 20
            flags.append("No saving or checking account records found")

        # Rule 4: Young applicant
        if age < 22:
            score += 15
            flags.append(f"Applicant age ({age:.0f}) below 22 — limited credit history")

        # Rule 5: Unskilled / no residence
        if job == 0:
            score += 10
            flags.append("Job class 0 (unskilled non-resident) — elevated instability risk")

        # Rule 6: Low income-to-loan ratio
        if i2l < 0.2:
            score += 15
            flags.append(f"Income-to-loan ratio ({i2l:.3f}) is very low — affordability concern")

        # Rule 7: High feature-engineered fraud score
        if fe_fraud_score >= 50:
            score += 20
            flags.append(f"Engineered fraud indicator score is high ({fe_fraud_score:.0f}/100)")

        score = min(score, 100.0)
        is_fraud = score >= FRAUD_SCORE_THRESHOLD

        if score >= 75:
            recommendation = "DECLINE — Very high fraud/default risk. Manual review required."
        elif score >= FRAUD_SCORE_THRESHOLD:
            recommendation = "REVIEW — Elevated risk detected. Escalate to senior analyst."
        elif score >= 30:
            recommendation = "CAUTION — Some risk indicators present. Proceed with additional verification."
        else:
            recommendation = "APPROVE — No significant fraud indicators detected."

        logger.info(f"Fraud score: {score:.1f} | Flagged: {is_fraud} | Flags: {len(flags)}")
        return FraudResult(
            fraud_score=round(score, 1),
            is_fraud_flag=is_fraud,
            flags=flags,
            recommendation=recommendation,
        )
