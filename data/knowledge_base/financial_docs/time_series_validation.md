# Financial Time-Series Validation & Leakage Prevention

## Chronological Data Splitting
Financial time-series data exhibits temporal autocorrelation and non-stationarity. Random cross-validation (e.g., `train_test_split`) causes severe lookahead bias and data leakage by training models on future data points to predict past data points.
Proper validation requires strict chronological splitting:
- Training set: Oldest 70% of observations.
- Validation set: Subsequent 15% of observations.
- Test set: Newest 15% of observations (evaluated only once after final model selection).

## Target Purging and Embargo
When target labels represent multi-day forward returns (e.g., 7-day future return `future_return_7d`), consecutive observations share overlapping future price windows.
To prevent target leakage between training and validation sets:
- **Purging**: Removes training samples near the end of the training set whose multi-day return target window overlaps with validation sample timestamps.
- **Embargoing**: Inserts a buffer period after validation boundaries to prevent post-validation leakage.

## Directional Accuracy vs Profitability
High directional accuracy (% of correct price direction predictions) does not automatically guarantee profitable trading performance due to asymmetric return magnitudes, transaction fees, slippage, and regime shifts. Models must be evaluated across RMSE, MAE, Information Coefficient (IC), and risk-adjusted metrics.
