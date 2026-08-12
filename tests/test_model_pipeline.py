"""
Unit tests for Machine Learning Model Pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from src.model_pipeline import FraudModelPipeline


def test_data_preparation_and_stratification():
    pipeline = FraudModelPipeline(test_size=0.25, random_seed=42)
    # Mock feature dataframe
    n_samples = 100
    df = pd.DataFrame({
        "provider_id": [f"P{i}" for i in range(n_samples)],
        "potential_fraud": [1 if i < 10 else 0 for i in range(n_samples)],
        "feat_1": np.random.randn(n_samples),
        "feat_2": np.random.uniform(0, 100, n_samples),
        "feat_3": np.random.randint(1, 10, n_samples),
    })
    X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
    assert len(X_train) == 75
    assert len(X_test) == 25
    assert "provider_id" not in X_train.columns
    assert "potential_fraud" not in X_train.columns
    # Check stratification ratio is maintained
    assert y_train.mean() == pytest.approx(0.10, abs=0.03)
    assert y_test.mean() == pytest.approx(0.10, abs=0.03)


def test_xgboost_training_and_threshold_selection():
    pipeline = FraudModelPipeline(test_size=0.30, random_seed=42)
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "provider_id": [f"P{i}" for i in range(n)],
        "potential_fraud": [1 if i < 20 else 0 for i in range(n)],
        "total_claims": np.concatenate([np.random.normal(500, 50, 20), np.random.normal(50, 10, 100)]),
        "mean_payout": np.concatenate([np.random.normal(2000, 200, 20), np.random.normal(300, 50, 100)]),
    })
    X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
    pipeline.train_xgboost(X_train, y_train)
    assert pipeline.model is not None

    threshold = pipeline.tune_threshold(X_train, y_train)
    assert 0.0 < threshold < 1.0

    eval_results = pipeline.evaluate_test_set(X_test, y_test)
    assert "roc_auc" in eval_results
    assert eval_results["roc_auc"] >= 0.80
