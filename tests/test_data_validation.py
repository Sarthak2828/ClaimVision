"""
Unit tests for ClaimVision Data Validation.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_validator import DataValidator, DataQualityError


def test_schema_validation_passes_valid_data():
    validator = DataValidator()
    dfs = {
        "labels": pd.DataFrame({"Provider": ["P1"], "PotentialFraud": ["Yes"]}),
        "beneficiary": pd.DataFrame({
            "BeneID": ["B1"], "DOB": ["1950-01-01"], "Gender": [1], "Race": [1], "State": [1], "County": [10]
        }),
        "inpatient": pd.DataFrame({
            "BeneID": ["B1"], "ClaimID": ["C1"], "ClaimStartDt": ["2009-01-01"], "ClaimEndDt": ["2009-01-05"],
            "Provider": ["P1"], "InscClaimAmtReimbursed": [5000.0], "AdmissionDt": ["2009-01-01"], "DischargeDt": ["2009-01-05"]
        }),
        "outpatient": pd.DataFrame({
            "BeneID": ["B1"], "ClaimID": ["C2"], "ClaimStartDt": ["2009-02-01"], "ClaimEndDt": ["2009-02-01"],
            "Provider": ["P1"], "InscClaimAmtReimbursed": [200.0]
        }),
    }
    assert validator.validate_schema(dfs) is True


def test_schema_validation_fails_missing_table():
    validator = DataValidator()
    dfs = {"labels": pd.DataFrame({"Provider": ["P1"], "PotentialFraud": ["No"]})}
    assert validator.validate_schema(dfs) is False
    assert any("Missing required table" in err for err in validator.report["errors"])


def test_target_leakage_audit_catches_forbidden_columns():
    validator = DataValidator()
    dfs = {
        "labels": pd.DataFrame({"Provider": ["P1"], "PotentialFraud": ["No"]}),
        "inpatient": pd.DataFrame({"ClaimID": ["C1"], "RecoveryAmount": [1000.0]}),
    }
    passed = validator.audit_target_leakage(dfs)
    assert passed is False
    assert any("Potential target leakage" in err for err in validator.report["errors"])


def test_negative_reimbursement_assertion():
    validator = DataValidator()
    dfs = {
        "labels": pd.DataFrame({"Provider": ["P1"], "PotentialFraud": ["No"]}),
        "inpatient": pd.DataFrame({
            "BeneID": ["B1"], "ClaimID": ["C1"], "ClaimStartDt": ["2009-01-01"], "ClaimEndDt": ["2009-01-05"],
            "Provider": ["P1"], "InscClaimAmtReimbursed": [-500.0], "AdmissionDt": ["2009-01-01"], "DischargeDt": ["2009-01-05"]
        }),
    }
    validator.validate_integrity_and_ranges(dfs)
    assert any("negative reimbursement" in err for err in validator.report["errors"])
