"""
Unit tests for provider-level feature engineering.
Tests requiring a live PostgreSQL connection are skipped automatically when the
database is not reachable.
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer


def make_synthetic_data():
    """Create synthetic in-memory data matching the relational schema."""
    providers = pd.DataFrame({
        "provider_id": ["P1", "P2", "P3"],
        "potential_fraud": [1, 0, 1],
    })
    beneficiaries = pd.DataFrame({
        "bene_id": ["B1", "B2", "B3"],
        "dob": ["1950-01-01", "1960-05-10", "1955-03-20"],
        "dod": [None, None, None],
        "gender": [1, 2, 1], "race": [1, 2, 1],
        "state_id": [1, 2, 1], "county_id": [10, 20, 10],
        "renal_disease": [False, True, False],
        "chronic_alzheimer": [True, False, False],
        "chronic_heartfailure": [False, True, False],
        "chronic_kidneydisease": [False, False, True],
        "chronic_cancer": [False, False, False],
        "chronic_copd": [False, True, False],
        "chronic_depression": [True, False, True],
        "chronic_diabetes": [True, True, False],
        "chronic_ischemicheart": [False, False, True],
        "chronic_osteoporosis": [False, True, False],
        "chronic_rheumatoidarthritis": [False, False, False],
        "chronic_stroke": [False, False, True],
        "ip_annual_reimbursement": [1000.0, 0.0, 500.0],
        "ip_annual_deductible": [100.0, 0.0, 50.0],
        "op_annual_reimbursement": [200.0, 500.0, 100.0],
        "op_annual_deductible": [50.0, 100.0, 25.0],
    })
    claims = pd.DataFrame({
        "claim_id": [f"C{i}" for i in range(10)],
        "claim_type": ["Inpatient"] * 3 + ["Outpatient"] * 7,
        "bene_id": ["B1", "B2", "B3", "B1", "B2", "B3", "B1", "B2", "B3", "B1"],
        "provider_id": ["P1", "P1", "P2", "P2", "P3", "P3", "P1", "P2", "P3", "P1"],
        "claim_start_date": ["2009-01-01"] * 10,
        "reimbursed_amount": [4500, 3000, 1000, 500, 700, 300, 200, 100, 800, 600],
        "deductible_amount": [1000, 500, 0, 50, 100, 0, 0, 0, 50, 0],
        "length_of_stay": [4, 5, 0, 0, 0, 0, 0, 0, 0, 0],
        "attending_physician": ["PHY1", "PHY1", "PHY2", "PHY2", "PHY3", None, "PHY1", "PHY2", None, "PHY1"],
        "operating_physician": [None] * 10,
        "primary_diagnosis_code": ["4100", "7800", "4019", "V5789", "4100", "4019", "7800", "4019", "4100", None],
    })
    inpatient = pd.DataFrame({
        "claim_id": ["C0", "C1", "C2"],
        "provider_id": ["P1", "P1", "P2"],
        "length_of_stay": [4, 5, 0],
        "primary_procedure_code": ["3600", "3600", None],
    })
    return {"providers": providers, "beneficiaries": beneficiaries, "claims": claims, "inpatient": inpatient}


def test_feature_engineering_in_memory():
    """Feature engineering must produce correct shape and zero target leakage."""
    fe = FeatureEngineer.__new__(FeatureEngineer)
    data = make_synthetic_data()
    df_features = fe.compute_features(data=data)

    assert "provider_id" in df_features.columns
    assert "potential_fraud" in df_features.columns
    assert len(df_features) == 3  # 3 providers

    # No target-derived columns in features
    forbidden = {"PotentialFraud", "fraud_label"}
    assert not forbidden.intersection(set(df_features.columns)), \
        f"Forbidden columns found: {forbidden.intersection(set(df_features.columns))}"

    # All financial aggregates must be non-negative
    assert (df_features["total_reimbursed"] >= 0).all()
    assert (df_features["total_claims"] >= 0).all()


def test_feature_engineering_no_leakage_columns():
    """
    No feature column derived from the target label should be present.
    'potential_fraud' is the target itself (expected, will be dropped by the ML pipeline).
    Leakage would be a column like 'fraud_score', 'is_flagged_by_rules', or
    'pct_claims_fraud' that encodes fraud-derived knowledge into the features.
    """
    fe = FeatureEngineer.__new__(FeatureEngineer)
    data = make_synthetic_data()
    df_features = fe.compute_features(data=data)

    # These columns would indicate actual target leakage into features
    forbidden_derived = {"fraud_label", "PotentialFraud", "is_fraud", "fraud_probability",
                         "fraud_score", "flagged_by_rules"}
    feature_cols = set(df_features.columns) - {"provider_id", "potential_fraud"}
    for col in feature_cols:
        assert col not in forbidden_derived, (
            f"Target-derived column '{col}' found in feature matrix — data leakage!"
        )


def test_feature_engineering_db():
    """Integration test: Feature engineering reads from live PostgreSQL database."""
    try:
        from database.db_manager import DatabaseManager
        DatabaseManager()
    except Exception as e:
        pytest.skip(f"PostgreSQL not available ({e}). Run: docker compose up -d")

    fe = FeatureEngineer()
    df = fe.compute_features()
    assert len(df) > 0
    assert "total_claims" in df.columns
    assert "potential_fraud" in df.columns
