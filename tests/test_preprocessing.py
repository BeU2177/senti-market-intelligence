"""Unit tests for FeaturePreprocessor fitting strictly on training set."""

import pytest
import pandas as pd
import numpy as np

from models.preprocessing import FeaturePreprocessor


def test_preprocessor_fits_exclusively_on_train():
    """Verify preprocessor mean/std are derived strictly from X_train."""
    X_train = pd.DataFrame({"feat_1": [10.0, 20.0, 30.0, 40.0], "feat_2": [1.0, 2.0, 3.0, 4.0]})
    X_val = pd.DataFrame({"feat_1": [100.0, 200.0], "feat_2": [10.0, 20.0]})

    prep = FeaturePreprocessor(impute_strategy="median", scale=True)
    X_tr_proc = prep.fit_transform(X_train)

    # Preprocessor mean should match X_train mean (25.0)
    assert np.isclose(prep.scaler.mean_[0], 25.0)

    # Transforming X_val using X_train fitted parameters
    X_v_proc = prep.transform(X_val)

    assert X_v_proc.shape == (2, 2)
    # Scaled X_val value for 100.0 relative to train mean (25.0) and std (~11.18)
    expected_scaled = (100.0 - prep.scaler.mean_[0]) / prep.scaler.scale_[0]
    assert np.isclose(X_v_proc[0, 0], expected_scaled)
