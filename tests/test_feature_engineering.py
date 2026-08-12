"""
Unit tests for Feature Engineering.
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer


def test_feature_engineering_computation():
    fe = FeatureEngineer()
    mock_data = {
        "providers": pd.DataFrame({"provider_id": ["P1", "P2"], "potential_fraud": [1, 0]}),
        "beneficiaries": pd.DataFrame({
            "bene_id": ["B1", "B2"], "dob": ["1950-01-01", "1960-01-01"], "gender": [1, 2],
            "race": [1, 2], "state_id": [1, 1], "county_id": [10, 10], "renal_disease": [True, False],
            "chronic_alzheimer": [True, False], "chronic_heartfailure": [True, False],
            "chronic_kidneydisease": [False, False], "chronic_cancer": [False, False],
            "chronic_copd": [False, False], "chronic_depression": [False, False],
            "chronic_diabetes": [True, False], "chronic_ischemicheart": [True, False],
            "chronic_osteoporosis": [False, False], "chronic_rheumatoidarthritis": [False, False],
            "chronic_stroke": [False, False]
        }),
        "claims": pd.DataFrame({
            "claim_id": ["C1", "C2", "C3"],
            "claim_type": ["Inpatient", "Outpatient", "Outpatient"],
            "bene_id": ["B1", "B1", "B2"],
            "provider_id": ["P1", "P1", "P2"],
            "claim_start_date": ["2009-01-01", "2009-01-01", "2009-02-01"],
            "reimbursed_amount": [5000.0, 500.0, 300.0],
            "deductible_amount": [1000.0, 50.0, 0.0],
            "length_of_stay": [5, 0, 0],
            "attending_physician": ["PHY1", "PHY1", "PHY2"],
            "operating_physician": ["PHY2", None, None],
            "primary_diagnosis_code": ["4100", "4100", "7800"],
        }),
        "inpatient": pd.DataFrame({
            "claim_id": ["C1"], "provider_id": ["P1"], "length_of_stay": [5], "primary_procedure_code": ["3600"]
        }),
    }
    features = fe.compute_features(mock_data)
    assert len(features) == 2
    assert "total_claims" in features.columns
    assert "inpatient_ratio" in features.columns
    assert "mean_reimbursed" in features.columns
    assert "same_day_duplicate_claims" in features.columns
    assert features.loc[features["provider_id"] == "P1", "total_claims"].values[0] == 2
    assert features.loc[features["provider_id"] == "P1", "same_day_duplicate_claims"].values[0] == 2
