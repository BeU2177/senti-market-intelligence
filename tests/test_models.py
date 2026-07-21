"""Unit tests for baselines and ModelRegistry estimators."""

import pytest
import numpy as np

from models.baselines import PersistenceBaseline, HistoricalMeanBaseline
from models.model_registry import ModelRegistry


def test_baselines_prediction():
    """Test Persistence and HistoricalMean baselines."""
    X_tr = np.random.normal(0, 1, (50, 5))
    y_tr = np.array([0.01, 0.02, -0.01, 0.03, -0.02] * 10)
    X_v = np.random.normal(0, 1, (10, 5))

    p_base = PersistenceBaseline()
    p_base.fit(X_tr, y_tr)
    p_preds = p_base.predict(X_v)
    assert len(p_preds) == 10
    assert np.all(p_preds == 0.0)

    m_base = HistoricalMeanBaseline()
    m_base.fit(X_tr, y_tr)
    m_preds = m_base.predict(X_v)
    assert len(m_preds) == 10
    assert np.all(np.isclose(m_preds, np.mean(y_tr)))


def test_model_registry_instantiations():
    """Test model registry returns valid trainable estimators."""
    registry = ModelRegistry(random_seed=42)
    available = registry.get_available_models()

    assert "persistence" in available
    assert "historical_mean" in available
    assert "ridge" in available
    assert "random_forest" in available

    X_tr = np.random.normal(0, 1, (30, 4))
    y_tr = np.random.normal(0, 0.02, 30)

    for m_name in ["ridge", "random_forest", "extra_trees", "gradient_boosting"]:
        est = registry.get_model(m_name)
        est.fit(X_tr, y_tr)
        preds = est.predict(X_tr)
        assert len(preds) == 30
