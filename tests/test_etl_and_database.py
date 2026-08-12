"""
Unit and integration tests for ETL star-schema transformation and database connectivity.
Tests requiring a live PostgreSQL connection are skipped automatically when the
database is not reachable (e.g., in CI without Docker Compose).
"""

import pytest
import pandas as pd
from src.etl_pipeline import ClaimsETLPipeline


# ── Fixture: conditionally skip when PostgreSQL is not reachable ───────────────
def db_or_skip():
    """Return a DatabaseManager or pytest.skip if PostgreSQL is not available."""
    try:
        from database.db_manager import DatabaseManager
        return DatabaseManager()
    except Exception as e:
        pytest.skip(f"PostgreSQL not available ({e}). Run: docker compose up -d")


# ── Test 1: Pure-Python star-schema transformation (no DB required) ────────────
def test_star_schema_transformation():
    """Transform raw DataFrames into star-schema tables without touching the DB."""
    pipeline = ClaimsETLPipeline.__new__(ClaimsETLPipeline)
    pipeline.validator = _make_noop_validator()

    raw_dfs = {
        "labels": pd.DataFrame({"Provider": ["P1", "P2"], "PotentialFraud": ["Yes", "No"]}),
        "beneficiary": pd.DataFrame({
            "BeneID": ["B1", "B2"], "DOB": ["1950-01-01", "1960-05-10"], "DOD": [None, None],
            "Gender": [1, 2], "Race": [1, 2], "State": [1, 2], "County": [10, 20],
            "RenalDiseaseIndicator": ["0", "Y"],
            "ChronicCond_Alzheimer": [1, 2], "ChronicCond_Heartfailure": [2, 1],
            "ChronicCond_KidneyDisease": [2, 2], "ChronicCond_Cancer": [2, 2],
            "ChronicCond_ObstrPulmonary": [2, 2], "ChronicCond_Depression": [2, 2],
            "ChronicCond_Diabetes": [1, 2], "ChronicCond_IschemicHeart": [1, 1],
            "ChronicCond_Osteoporasis": [2, 2], "ChronicCond_rheumatoidarthritis": [2, 2],
            "ChronicCond_stroke": [2, 2],
            "IPAnnualReimbursementAmt": [1000.0, 0.0], "IPAnnualDeductibleAmt": [100.0, 0.0],
            "OPAnnualReimbursementAmt": [200.0, 500.0], "OPAnnualDeductibleAmt": [50.0, 100.0],
        }),
        "inpatient": pd.DataFrame({
            "BeneID": ["B1"], "ClaimID": ["C1"],
            "ClaimStartDt": ["2009-01-01"], "ClaimEndDt": ["2009-01-05"],
            "Provider": ["P1"], "InscClaimAmtReimbursed": [4500.0], "DeductibleAmtPaid": [1000.0],
            "AdmissionDt": ["2009-01-01"], "DischargeDt": ["2009-01-05"],
            "AttendingPhysician": ["PHY1"], "OperatingPhysician": ["PHY2"], "OtherPhysician": [None],
            "ClmAdmitDiagnosisCode": ["4100"], "DiagnosisGroupCode": ["100"],
            "ClmDiagnosisCode_1": ["4100"], "ClmProcedureCode_1": ["3600"],
        }),
        "outpatient": pd.DataFrame({
            "BeneID": ["B2"], "ClaimID": ["C2"],
            "ClaimStartDt": ["2009-02-01"], "ClaimEndDt": ["2009-02-01"],
            "Provider": ["P2"], "InscClaimAmtReimbursed": [350.0], "DeductibleAmtPaid": [50.0],
            "AttendingPhysician": ["PHY3"], "OperatingPhysician": [None], "OtherPhysician": [None],
            "ClmDiagnosisCode_1": ["7800"], "ClmProcedureCode_1": [None],
        }),
    }
    transformed = pipeline.transform(raw_dfs)
    assert len(transformed["dim_providers"]) == 2
    assert len(transformed["dim_beneficiaries"]) == 2
    assert len(transformed["fact_inpatient_claims"]) == 1
    assert len(transformed["fact_outpatient_claims"]) == 1
    assert len(transformed["fact_claims_unified"]) == 2
    assert transformed["fact_inpatient_claims"]["length_of_stay"].iloc[0] == 4


def test_database_manager_connection():
    """Verify DatabaseManager connects successfully to PostgreSQL."""
    db = db_or_skip()
    assert db.engine is not None
    counts = db.get_table_counts()
    assert "dim_providers" in counts
    assert "fact_claims_unified" in counts


# ── Helpers ───────────────────────────────────────────────────────────────────
class _NoopValidator:
    def run_all_checks(self, dfs):
        pass

def _make_noop_validator():
    return _NoopValidator()
