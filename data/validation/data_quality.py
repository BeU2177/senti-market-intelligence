"""Expanded data quality validation rules and freshness classification for market data."""

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from data.schemas.market_data import DataValidationResult, FreshnessStatus

REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "data_source",
]


class DataQualityValidator:
    """Validator performing structural integrity checks, market gap analysis, and freshness classification."""

    def __init__(self, required_columns: List[str] = REQUIRED_COLUMNS):
        self.required_columns = required_columns

    def validate(self, df: pd.DataFrame, symbol: str = "UNKNOWN", interval: str = "1d") -> DataValidationResult:
        """Execute comprehensive quality checks on market data DataFrame."""
        errors: List[str] = []
        warnings: List[str] = []

        if df is None or df.empty:
            return DataValidationResult(
                is_valid=False,
                symbol=symbol,
                errors=["DataFrame is empty or None"],
                warnings=[],
                row_count=0,
                missing_values=0,
                duplicate_timestamps=0,
                freshness_status=FreshnessStatus.UNKNOWN,
            )

        row_count = len(df)

        # 1. Required Columns Check
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        if "timestamp" not in df.columns or "close" not in df.columns:
            return DataValidationResult(
                is_valid=False,
                symbol=symbol,
                errors=errors,
                warnings=warnings,
                row_count=row_count,
                missing_values=int(df.isna().sum().sum()),
                duplicate_timestamps=0,
                freshness_status=FreshnessStatus.UNKNOWN,
            )

        # 2. Missing Values Check
        valid_cols = [c for c in self.required_columns if c in df.columns]
        total_missing = int(df[valid_cols].isna().sum().sum())
        if total_missing > 0:
            warnings.append(f"Found {total_missing} missing value(s) across required fields")

        # 3. Duplicate Timestamps Check
        duplicate_ts_count = int(df.duplicated(subset=["timestamp"]).sum())
        if duplicate_ts_count > 0:
            errors.append(f"Found {duplicate_ts_count} duplicate timestamp(s)")

        # 4. Timestamp Ordering Check
        is_sorted = df["timestamp"].is_monotonic_increasing
        if not is_sorted:
            warnings.append("Timestamps are not strictly monotonically increasing")

        # 5. Price Sanity Checks
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        negative_prices = 0
        for col in price_cols:
            neg_c = int((df[col] < 0).sum())
            if neg_c > 0:
                negative_prices += neg_c
                errors.append(f"Column '{col}' contains {neg_c} negative price value(s)")

            zero_c = int((df[col] == 0).sum())
            if zero_c > 0:
                warnings.append(f"Column '{col}' contains {zero_c} zero price value(s)")

        # 6. OHLC Relationship Checks
        ohlc_errors = 0
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            high_low_inv = int((df["high"] < df["low"]).sum())
            high_open_inv = int((df["high"] < df["open"]).sum())
            high_close_inv = int((df["high"] < df["close"]).sum())
            low_open_inv = int((df["low"] > df["open"]).sum())
            low_close_inv = int((df["low"] > df["close"]).sum())

            ohlc_errors = high_low_inv + high_open_inv + high_close_inv + low_open_inv + low_close_inv
            if high_low_inv > 0:
                errors.append(f"Found {high_low_inv} row(s) where High < Low")
            if high_open_inv > 0:
                errors.append(f"Found {high_open_inv} row(s) where High < Open")
            if high_close_inv > 0:
                errors.append(f"Found {high_close_inv} row(s) where High < Close")
            if low_open_inv > 0:
                errors.append(f"Found {low_open_inv} row(s) where Low > Open")
            if low_close_inv > 0:
                errors.append(f"Found {low_close_inv} row(s) where Low > Close")

        # 7. Volume Sanity Check
        negative_vol_count = 0
        if "volume" in df.columns:
            negative_vol_count = int((df["volume"] < 0).sum())
            if negative_vol_count > 0:
                errors.append(f"Column 'volume' contains {negative_vol_count} negative volume value(s)")

        # 8. Earliest & Latest Timestamps
        earliest_ts = df["timestamp"].min()
        latest_ts = df["timestamp"].max()
        
        earliest_dt = earliest_ts.to_pydatetime() if hasattr(earliest_ts, "to_pydatetime") else earliest_ts
        latest_dt = latest_ts.to_pydatetime() if hasattr(latest_ts, "to_pydatetime") else latest_ts

        # 9. Freshness Classification
        freshness = self._classify_freshness(latest_dt)

        # 10. Market Gap Detection (Weekend-aware)
        data_gaps = self._detect_data_gaps(df, interval)
        if data_gaps:
            warnings.append(f"Detected {len(data_gaps)} unusual data gap(s) exceeding normal market trading breaks")

        is_valid = len(errors) == 0

        return DataValidationResult(
            is_valid=is_valid,
            symbol=symbol,
            errors=errors,
            warnings=warnings,
            row_count=row_count,
            missing_values=total_missing,
            duplicate_timestamps=duplicate_ts_count,
            negative_prices=negative_prices,
            negative_volume=negative_vol_count,
            ohlc_errors=ohlc_errors,
            data_gaps=data_gaps,
            earliest_timestamp=earliest_dt,
            latest_timestamp=latest_dt,
            freshness_status=freshness,
            details={
                "checked_columns": self.required_columns,
                "null_counts": df[valid_cols].isna().sum().to_dict(),
            },
        )

    def _classify_freshness(self, latest_dt: Optional[datetime]) -> FreshnessStatus:
        """Classify dataset freshness based on age relative to current UTC time."""
        if not latest_dt:
            return FreshnessStatus.UNKNOWN

        now = datetime.now(timezone.utc)
        if latest_dt.tzinfo is None:
            latest_dt = latest_dt.replace(tzinfo=timezone.utc)

        age = now - latest_dt

        if age < timedelta(minutes=15):
            return FreshnessStatus.NEAR_REAL_TIME
        elif age < timedelta(hours=24):
            return FreshnessStatus.DELAYED
        elif age < timedelta(days=7):
            return FreshnessStatus.HISTORICAL
        else:
            return FreshnessStatus.STALE

    def _detect_data_gaps(self, df: pd.DataFrame, interval: str) -> List[Dict[str, Any]]:
        """Detect gaps in timestamps larger than expected trading interval (ignoring normal weekends)."""
        gaps = []
        if len(df) < 2 or "timestamp" not in df.columns:
            return gaps

        ts_series = pd.to_datetime(df["timestamp"]).sort_values().reset_index(drop=True)
        diffs = ts_series.diff()

        # Thresholds based on interval
        if interval in ["1d", "1wk"]:
            gap_threshold = timedelta(days=4)  # 4+ days skips Friday->Monday weekend cleanly
        elif interval in ["1h", "60m"]:
            gap_threshold = timedelta(days=4)
        else:
            gap_threshold = timedelta(days=4)

        gap_indices = diffs[diffs > gap_threshold].index

        for idx in gap_indices[:10]:  # Limit to top 10 gaps
            start_ts = ts_series.iloc[idx - 1]
            end_ts = ts_series.iloc[idx]
            gap_days = round((end_ts - start_ts).total_seconds() / 86400, 1)
            gaps.append({
                "start": start_ts.isoformat(),
                "end": end_ts.isoformat(),
                "gap_days": gap_days,
            })

        return gaps
