"""Automated lookahead bias and target leakage validation module."""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class LeakageValidationReport(BaseModel):
    """Structured report of data leakage validation checks."""
    is_clean: bool
    leakage_errors: List[str] = Field(default_factory=list)
    leakage_warnings: List[str] = Field(default_factory=list)
    checked_features_count: int = 0
    checked_targets_count: int = 0
    details: Dict[str, Any] = Field(default_factory=dict)


class LeakageValidator:
    """Validator inspecting feature DataFrames for lookahead bias and target leakage."""

    def __init__(self, target_prefix: str = "future_"):
        self.target_prefix = target_prefix

    def validate_dataset(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
    ) -> LeakageValidationReport:
        """Executes automated leakage checks on features and targets."""
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Target Column Isolation Check
        for f_col in feature_cols:
            if f_col.startswith(self.target_prefix):
                errors.append(f"Target column '{f_col}' found in feature column list!")
            if f_col in target_cols:
                errors.append(f"Overlap detected: '{f_col}' exists in both feature and target columns!")

        # 2. Check for Future Lookahead in Feature Values
        # If a feature at index t perfectly matches close[t+1] or open[t+1], flag as leakage
        if "close" in df.columns and len(df) > 10:
            future_close_1d = df["close"].shift(-1)
            for f_col in feature_cols:
                if f_col in df.columns and pd.api.types.is_numeric_dtype(df[f_col]):
                    series = df[f_col].dropna()
                    if len(series) > 10:
                        # Correlation check against future 1d close
                        valid_mask = series.index.isin(future_close_1d.dropna().index)
                        if valid_mask.sum() > 10:
                            s_clean = series[valid_mask]
                            fc_clean = future_close_1d.loc[s_clean.index]
                            corr = s_clean.corr(fc_clean)
                            
                            # If correlation with future close is extremely high (> 0.999) while feature is not identical to current close
                            curr_corr = s_clean.corr(df.loc[s_clean.index, "close"])
                            if not np.isnan(corr) and abs(corr) > 0.999 and (np.isnan(curr_corr) or abs(curr_corr) < 0.999):
                                errors.append(f"Feature '{f_col}' shows suspicious correlation ({corr:.4f}) with future 1d close price!")

        is_clean = len(errors) == 0

        return LeakageValidationReport(
            is_clean=is_clean,
            leakage_errors=errors,
            leakage_warnings=warnings,
            checked_features_count=len(feature_cols),
            checked_targets_count=len(target_cols),
            details={
                "target_prefix": self.target_prefix,
                "feature_count": len(feature_cols),
                "target_count": len(target_cols),
            },
        )
