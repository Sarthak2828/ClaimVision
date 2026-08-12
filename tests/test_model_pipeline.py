"""
Unit tests for ClaimVision ML pipeline.

Verifies:
- Provider-level group split (no provider appears in both train and test)
- Approximately 80/20 provider-level split ratio
- Fraud coverage in both splits
- Dynamic scale_pos_weight from training providers only
- Model trains and produces valid metrics
"""

import pytest
import pandas as pd
import numpy as np
from src.model_pipeline import FraudModelPipeline, provider_group_split


def make_fake_provider_features(n_providers=200, fraud_rate=0.10, seed=42):
    """Generate a synthetic provider-level feature DataFrame for testing."""
    rng = np.random.default_rng(seed)
    n_fraud = max(2, int(n_providers * fraud_rate))
    n_clean = n_providers - n_fraud
    labels = [1] * n_fraud + [0] * n_clean
    rng.shuffle(labels)
    return pd.DataFrame({
        "provider_id":                 [f"P{i:04d}" for i in range(n_providers)],
        "potential_fraud":             labels,
        "total_claims":                rng.integers(10, 2000, n_providers).tolist(),
        "total_reimbursed":            rng.uniform(1000, 5_000_000, n_providers).tolist(),
        "mean_reimbursed":             rng.uniform(100, 10000, n_providers).tolist(),
        "std_reimbursed":              rng.uniform(0, 5000, n_providers).tolist(),
        "max_reimbursed":              rng.uniform(1000, 125000, n_providers).tolist(),
        "median_reimbursed":           rng.uniform(100, 5000, n_providers).tolist(),
        "total_deductible":            rng.uniform(0, 500000, n_providers).tolist(),
        "mean_deductible":             rng.uniform(0, 1000, n_providers).tolist(),
        "zero_reimbursed_count":       rng.integers(0, 50, n_providers).tolist(),
        "inpatient_claims":            rng.integers(0, 500, n_providers).tolist(),
        "outpatient_claims":           rng.integers(0, 2000, n_providers).tolist(),
        "inpatient_ratio":             rng.uniform(0, 1, n_providers).tolist(),
        "deductible_to_reimbursement_ratio": rng.uniform(0, 1, n_providers).tolist(),
        "zero_reimbursed_ratio":       rng.uniform(0, 0.5, n_providers).tolist(),
        "mean_length_of_stay":         rng.uniform(0, 20, n_providers).tolist(),
        "max_length_of_stay":          rng.uniform(0, 60, n_providers).tolist(),
        "distinct_procedures":         rng.integers(0, 50, n_providers).tolist(),
        "distinct_attending_physicians": rng.integers(0, 100, n_providers).tolist(),
        "distinct_operating_physicians": rng.integers(0, 50, n_providers).tolist(),
        "distinct_diagnosis_codes":    rng.integers(1, 200, n_providers).tolist(),
        "distinct_beneficiaries":      rng.integers(1, 1000, n_providers).tolist(),
        "mean_patient_age":            rng.uniform(50, 85, n_providers).tolist(),
        "renal_disease_ratio":         rng.uniform(0, 1, n_providers).tolist(),
        "mean_chronic_conditions":     rng.uniform(0, 8, n_providers).tolist(),
        "alzheimer_ratio":             rng.uniform(0, 1, n_providers).tolist(),
        "heartfailure_ratio":          rng.uniform(0, 1, n_providers).tolist(),
        "kidneydisease_ratio":         rng.uniform(0, 1, n_providers).tolist(),
        "diabetes_ratio":              rng.uniform(0, 1, n_providers).tolist(),
        "ischemic_heart_ratio":        rng.uniform(0, 1, n_providers).tolist(),
        "same_day_duplicate_claims":   rng.integers(0, 50, n_providers).tolist(),
        "claims_per_beneficiary":      rng.uniform(1, 5, n_providers).tolist(),
        "physician_to_claim_ratio":    rng.uniform(0, 1, n_providers).tolist(),
        "reimbursement_per_beneficiary": rng.uniform(100, 50000, n_providers).tolist(),
    })


def test_provider_group_split_disjoint():
    """Critical: No provider may appear in both train and test sets."""
    df = make_fake_provider_features(n_providers=200)
    _, _, _, _, train_ids, test_ids = provider_group_split(df)
    assert set(train_ids).isdisjoint(set(test_ids)), (
        "Provider IDs overlap between train and test — data leakage!"
    )


def test_provider_group_split_sizes():
    """Provider-level test fraction should be approximately 20%."""
    df = make_fake_provider_features(n_providers=200)
    _, _, _, _, train_ids, test_ids = provider_group_split(df, test_size=0.20)
    test_frac = len(test_ids) / (len(train_ids) + len(test_ids))
    assert 0.15 <= test_frac <= 0.25, f"Test fraction {test_frac:.2f} outside expected range"


def test_provider_group_split_both_have_fraud():
    """Both train and test must contain fraud providers."""
    df = make_fake_provider_features(n_providers=200, fraud_rate=0.15)
    _, _, y_train, y_test, _, _ = provider_group_split(df)
    assert y_train.sum() > 0, "No fraud providers in train set"
    assert y_test.sum() > 0, "No fraud providers in test set"


def test_pipeline_provider_disjoint():
    """FraudModelPipeline.prepare_data must enforce provider-level disjoint split."""
    df = make_fake_provider_features(n_providers=200)
    pipeline = FraudModelPipeline(random_seed=42, test_size=0.20)
    pipeline.prepare_data(df)
    assert set(pipeline.train_provider_ids).isdisjoint(set(pipeline.test_provider_ids)), (
        "Pipeline train/test provider IDs must be disjoint"
    )


def test_scale_pos_weight_is_dynamic():
    """scale_pos_weight must be calculated dynamically from training data only."""
    df = make_fake_provider_features(n_providers=200, fraud_rate=0.10)
    pipeline = FraudModelPipeline(random_seed=42, n_folds=2)
    X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
    pipeline.train_xgboost(X_train, y_train)

    expected_spw = float((y_train == 0).sum()) / float((y_train == 1).sum())
    stored_spw = pipeline.metrics["scale_pos_weight"]
    assert abs(stored_spw - round(expected_spw, 4)) < 0.001, (
        f"scale_pos_weight mismatch: stored={stored_spw}, expected={round(expected_spw, 4)}"
    )


def test_pipeline_full_run():
    """Full pipeline run on synthetic provider data must produce valid metrics."""
    df = make_fake_provider_features(n_providers=300, fraud_rate=0.10)
    pipeline = FraudModelPipeline(random_seed=42, n_folds=2)
    metrics = pipeline.run(df_features=df, save_artifacts=False)
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert "tuned_threshold" in metrics
    assert "scale_pos_weight" in pipeline.metrics
