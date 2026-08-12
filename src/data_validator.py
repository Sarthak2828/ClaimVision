"""
ClaimVision Data Validation Module.
Performs data quality checks, schema validation, range checks, and leakage audits.
"""

import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DataQualityError(Exception):
    """Raised when critical data quality checks fail."""
    pass


class DataValidator:
    """Validates healthcare claims datasets against domain rules and schema contracts."""

    REQUIRED_COLUMNS = {
        "labels": ["Provider", "PotentialFraud"],
        "beneficiary": ["BeneID", "DOB", "Gender", "Race", "State", "County"],
        "inpatient": ["BeneID", "ClaimID", "ClaimStartDt", "ClaimEndDt", "Provider", 
                      "InscClaimAmtReimbursed", "AdmissionDt", "DischargeDt"],
        "outpatient": ["BeneID", "ClaimID", "ClaimStartDt", "ClaimEndDt", "Provider", 
                       "InscClaimAmtReimbursed"],
    }

    LEAKAGE_COLUMNS = [
        "AuditResult", "RecoveryAmount", "InvestigationStatus", "ProsecutionDate",
        "SettlementAmount", "FraudOutcome", "FraudConfidenceScore"
    ]

    def __init__(self, fail_fast: bool = False):
        self.fail_fast = fail_fast
        self.report: Dict[str, Any] = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

    def validate_schema(self, dfs: Dict[str, pd.DataFrame]) -> bool:
        """Verify presence of all mandatory tables and columns."""
        for table_name, req_cols in self.REQUIRED_COLUMNS.items():
            if table_name not in dfs:
                msg = f"Missing required table: '{table_name}'"
                self.report["errors"].append(msg)
                self.report["passed"] = False
                continue

            df = dfs[table_name]
            missing_cols = [c for c in req_cols if c not in df.columns]
            if missing_cols:
                msg = f"Table '{table_name}' missing required columns: {missing_cols}"
                self.report["errors"].append(msg)
                self.report["passed"] = False
        return len(self.report["errors"]) == 0

    def audit_target_leakage(self, dfs: Dict[str, pd.DataFrame]) -> bool:
        """
        Ensure no post-investigation or target-derived columns exist in raw claims.
        Claims must strictly contain attributes observable at the moment of billing.
        """
        for table_name, df in dfs.items():
            found_leakage = [c for c in df.columns if c in self.LEAKAGE_COLUMNS]
            if found_leakage:
                msg = f"CRITICAL: Potential target leakage detected in '{table_name}': {found_leakage}"
                self.report["errors"].append(msg)
                self.report["passed"] = False

        # Verify Provider labels are binary and not embedded as numeric continuous ground-truth
        if "labels" in dfs:
            unique_targets = set(dfs["labels"]["PotentialFraud"].dropna().unique())
            valid_targets = {"Yes", "No", 0, 1}
            if not unique_targets.issubset(valid_targets):
                msg = f"Invalid target values in labels: {unique_targets}"
                self.report["errors"].append(msg)
                self.report["passed"] = False
        return self.report["passed"]

    def validate_integrity_and_ranges(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Perform domain-specific relational and range assertions."""
        labels = dfs.get("labels")
        bene = dfs.get("beneficiary")
        inp = dfs.get("inpatient")
        out = dfs.get("outpatient")

        # 1. Target distribution check
        if labels is not None:
            n_providers = len(labels)
            n_duplicates = labels["Provider"].duplicated().sum()
            if n_duplicates > 0:
                self.report["errors"].append(f"Found {n_duplicates} duplicate Provider IDs in labels.")
                self.report["passed"] = False

            fraud_counts = labels["PotentialFraud"].value_counts().to_dict()
            fraud_rate = labels["PotentialFraud"].map({"Yes": 1, "No": 0}).mean()
            self.report["metrics"]["provider_count"] = n_providers
            self.report["metrics"]["fraud_distribution"] = fraud_counts
            self.report["metrics"]["fraud_rate_pct"] = round(fraud_rate * 100, 2)
            logger.info(f"Target distribution: {fraud_counts} (Fraud Rate: {self.report['metrics']['fraud_rate_pct']}%)")

        # 2. Beneficiary demographics & date assertions
        if bene is not None:
            self.report["metrics"]["beneficiary_count"] = len(bene)
            dob_dt = pd.to_datetime(bene["DOB"], errors="coerce")
            future_dobs = (dob_dt > pd.Timestamp.now()).sum()
            if future_dobs > 0:
                self.report["errors"].append(f"Beneficiary DOB contains {future_dobs} future dates.")
                self.report["passed"] = False

            # Check DOD >= DOB where DOD is present
            dod_present = bene["DOD"].notna()
            if dod_present.any():
                dod_dt = pd.to_datetime(bene.loc[dod_present, "DOD"], errors="coerce")
                dob_subset = dob_dt[dod_present]
                invalid_deaths = (dod_dt < dob_subset).sum()
                if invalid_deaths > 0:
                    self.report["errors"].append(f"Beneficiary DOD occurs before DOB in {invalid_deaths} records.")
                    self.report["passed"] = False

        # 3. Inpatient claims range assertions
        if inp is not None:
            self.report["metrics"]["inpatient_claims_count"] = len(inp)
            neg_reimb = (inp["InscClaimAmtReimbursed"] < 0).sum()
            if neg_reimb > 0:
                self.report["errors"].append(f"Inpatient has {neg_reimb} negative reimbursement claims.")
                self.report["passed"] = False

            # Check DischargeDt >= AdmissionDt
            adm_dt = pd.to_datetime(inp["AdmissionDt"], errors="coerce")
            dis_dt = pd.to_datetime(inp["DischargeDt"], errors="coerce")
            invalid_stays = (dis_dt < adm_dt).sum()
            if invalid_stays > 0:
                self.report["warnings"].append(f"Inpatient has {invalid_stays} records where DischargeDt < AdmissionDt.")

            # Check claim uniqueness
            inp_dup_claims = inp["ClaimID"].duplicated().sum()
            if inp_dup_claims > 0:
                self.report["warnings"].append(f"Inpatient contains {inp_dup_claims} duplicate ClaimIDs.")

        # 4. Outpatient claims range assertions
        if out is not None:
            self.report["metrics"]["outpatient_claims_count"] = len(out)
            neg_reimb = (out["InscClaimAmtReimbursed"] < 0).sum()
            if neg_reimb > 0:
                self.report["errors"].append(f"Outpatient has {neg_reimb} negative reimbursement claims.")
                self.report["passed"] = False

        if self.fail_fast and not self.report["passed"]:
            raise DataQualityError(f"Data validation failed: {self.report['errors']}")

        return self.report

    def run_all_checks(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Execute the entire validation pipeline."""
        logger.info("Executing schema validation...")
        self.validate_schema(dfs)

        logger.info("Executing target leakage audit...")
        self.audit_target_leakage(dfs)

        logger.info("Executing relational integrity and domain range checks...")
        self.validate_integrity_and_ranges(dfs)

        status_str = "PASSED" if self.report["passed"] else "FAILED"
        logger.info(f"Validation completed with status: {status_str} "
                    f"(Errors: {len(self.report['errors'])}, Warnings: {len(self.report['warnings'])})")
        return self.report


if __name__ == "__main__":
    from data_ingestion import ingest_data
    dfs = ingest_data(sample_mode=True, sample_provider_count=200)
    validator = DataValidator(fail_fast=True)
    report = validator.run_all_checks(dfs)
    print("Report Summary:", report["metrics"])
