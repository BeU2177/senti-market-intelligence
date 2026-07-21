# Machine Learning & Deep Learning Pipeline — Senti Market Intelligence

## Time-Series Split & Leakage Prevention

1. **Chronological Data Splitting**: Datasets are partitioned into 70% Training, 15% Validation, and 15% Test sets strictly chronologically without random shuffling.
2. **Preprocessing Isolation**: Missing value imputation (`SimpleImputer`) and feature scaling (`StandardScaler`) are fitted strictly on `X_train`.
3. **Sequence Window Boundaries**: 3D sequence tensors $[N, sequence\_length, feature\_count]$ are constructed after splitting dataframes to prevent boundary leakage across splits.

## Model Family Suite

### Classical Estimators
- `LinearRegression`, `Ridge`
- `RandomForestRegressor`, `ExtraTreesRegressor`
- `GradientBoostingRegressor`, `XGBoostRegressor`

### PyTorch Deep Learning Suite
- **MLP (`PyTorchMLP`)**: Multi-Layer Perceptron with Batch Normalization and Dropout.
- **LSTM (`PyTorchLSTM`)**: Multi-layer unidirectional LSTM preserving temporal causality.
- **Temporal CNN (`PyTorchTemporalCNN`)**: 1D Dilated Causal Convolutional Neural Network with residual connections.
- **Transformer (`PyTorchTemporalTransformer`)**: Multi-Head Self-Attention model with positional encoding.

## Multi-Horizon Target Generation

Four forward return targets are predicted simultaneously:
- $R_{1d} = \frac{P_{t+1} - P_t}{P_t}$
- $R_{3d} = \frac{P_{t+3} - P_t}{P_t}$
- $R_{5d} = \frac{P_{t+5} - P_t}{P_t}$
- $R_{7d} = \frac{P_{t+7} - P_t}{P_t}$

Predicted target prices are calculated using:
$$P_{\text{pred}, h} = P_{\text{current}} \times (1 + R_{\text{pred}, h})$$

## Ensemble Blending & Confidence Scoring

Predictions across Classical ML and PyTorch models are combined using performance-weighted inverse validation RMSE:
$$w_m = \frac{1 / \text{RMSE}_m}{\sum_k 1 / \text{RMSE}_k}$$

Confidence is assessed non-fabricated:
- `HIGH`: Sign agreement $\ge 75\%$ and prediction std dev $\le 0.015$.
- `MEDIUM`: Sign agreement $\ge 50\%$ and prediction std dev $\le 0.030$.
- `LOW`: High prediction dispersion or conflicting signals.
