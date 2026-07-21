# MLOps Monitoring, Drift Detection & Model Lifecycle Management

## Production Prediction Logging

Every real-time prediction request processed by `InferenceService` is logged to `artifacts/predictions/predictions.jsonl` containing:
- Prediction ID
- Ticker Symbol & Timestamp
- Data Freshness & Latency Metrics
- Multi-Horizon Predicted Returns ($1d, 3d, 5d, 7d$) & Converted Target Prices
- Ensemble Member Weights
- Non-fabricated Confidence Level

## Realized Outcome Tracking

When target horizons expire, `MLOpsService` evaluates actual realized price returns against predicted returns, logging metrics to `artifacts/predictions/outcomes.jsonl`:
- Absolute Error ($AE = |R_{\text{pred}} - R_{\text{actual}}|$)
- Directional Correctness (`is_direction_correct`: boolean)

## Data & Feature Drift Monitoring

`MLOpsService.calculate_feature_drift` monitors statistical mean and standard deviation shifts between training baselines and production inputs:
$$\text{drift\_detected} = | \mu_{\text{production}} - \mu_{\text{training}} | > 2.0 \cdot \sigma_{\text{training}}$$

## 4-Stage Model Promotion Lifecycle

Models progress through 4 lifecycle stages:
1. **`CANDIDATE`**: Newly trained candidate model artifact.
2. **`VALIDATED`**: Evaluated on validation split.
3. **`APPROVED`**: Exceeds baseline performance without feature schema degradation.
4. **`PRODUCTION`**: Active inference model artifact serving real-time requests.
