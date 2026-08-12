"""
ClaimVision Feature Engineering Module.
Extracts leakage-free provider behavioral profiles, clinical severity indices,
and physician network metrics from normalized claims and beneficiary tables.
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from database.db_manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Computes leak-free provider-level feature vectors for fraud classification."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    def load_data_from_db(self) -> Dict[str, pd.DataFrame]:
        """Load relational tables needed for feature aggregation from database."""
        logger.info("Extracting relational tables from analytical database...")
        with self.db.engine.connect() as conn:
            providers = pd.read_sql("SELECT provider_id, potential_fraud FROM dim_providers", conn)
            beneficiaries = pd.read_sql("SELECT * FROM dim_beneficiaries", conn)
            claims = pd.read_sql("""
                SELECT 
                    claim_id, claim_type, bene_id, provider_id, claim_start_date,
                    reimbursed_amount, deductible_amount, length_of_stay,
                    attending_physician, operating_physician, primary_diagnosis_code
                FROM fact_claims_unified
            """, conn)
            inpatient = pd.read_sql("""
                SELECT claim_id, provider_id, length_of_stay, primary_procedure_code
                FROM fact_inpatient_claims
            """, conn)
            
        logger.info(f"Loaded: Providers={len(providers)}, Beneficiaries={len(beneficiaries)}, Claims={len(claims)}")
        return {
            "providers": providers,
            "beneficiaries": beneficiaries,
            "claims": claims,
            "inpatient": inpatient,
        }

    def compute_features(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Engineers provider-level feature vectors.
        All features are strictly derived from billing/demographic attributes
        available at claim submission time to prevent target leakage.
        """
        if data is None:
            data = self.load_data_from_db()

        providers = data["providers"].copy()
        beneficiaries = data["beneficiaries"].copy()
        claims = data["claims"].copy()
        inpatient = data["inpatient"].copy()

        logger.info("Computing provider billing and financial aggregations...")
        # 1. Financial & Volume Features per Provider
        fin_agg = claims.groupby("provider_id").agg(
            total_claims=("claim_id", "count"),
            inpatient_claims=("claim_type", lambda x: (x == "Inpatient").sum()),
            outpatient_claims=("claim_type", lambda x: (x == "Outpatient").sum()),
            total_reimbursed=("reimbursed_amount", "sum"),
            mean_reimbursed=("reimbursed_amount", "mean"),
            std_reimbursed=("reimbursed_amount", "std"),
            max_reimbursed=("reimbursed_amount", "max"),
            median_reimbursed=("reimbursed_amount", "median"),
            total_deductible=("deductible_amount", "sum"),
            mean_deductible=("deductible_amount", "mean"),
            zero_reimbursed_count=("reimbursed_amount", lambda x: (x == 0).sum()),
        ).reset_index()

        fin_agg["std_reimbursed"] = fin_agg["std_reimbursed"].fillna(0.0)
        fin_agg["inpatient_ratio"] = fin_agg["inpatient_claims"] / fin_agg["total_claims"]
        fin_agg["deductible_to_reimbursement_ratio"] = (
            fin_agg["total_deductible"] / (fin_agg["total_reimbursed"] + 1.0)
        )
        fin_agg["zero_reimbursed_ratio"] = fin_agg["zero_reimbursed_count"] / fin_agg["total_claims"]

        logger.info("Computing provider clinical and length-of-stay metrics...")
        # 2. Inpatient Length-of-Stay & Clinical Metrics
        if len(inpatient) > 0:
            inp_agg = inpatient.groupby("provider_id").agg(
                mean_length_of_stay=("length_of_stay", "mean"),
                max_length_of_stay=("length_of_stay", "max"),
                distinct_procedures=("primary_procedure_code", lambda x: x.dropna().nunique()),
            ).reset_index()
        else:
            inp_agg = pd.DataFrame(columns=["provider_id", "mean_length_of_stay", "max_length_of_stay", "distinct_procedures"])

        # 3. Physician Multiplicity & Network Diversity
        logger.info("Computing physician network complexity metrics...")
        phys_agg = claims.groupby("provider_id").agg(
            distinct_attending_physicians=("attending_physician", lambda x: x.dropna().nunique()),
            distinct_operating_physicians=("operating_physician", lambda x: x.dropna().nunique()),
            distinct_diagnosis_codes=("primary_diagnosis_code", lambda x: x.dropna().nunique()),
        ).reset_index()

        # 4. Patient Demographics & Chronic Disease Prevalence per Provider
        logger.info("Joining patient demographics and calculating chronic disease indices...")
        claims_bene = claims[["provider_id", "bene_id"]].drop_duplicates().merge(
            beneficiaries, on="bene_id", how="left"
        )
        # Compute age in years (reference year 2009 for CMS Medicare)
        bene_birth_year = pd.to_datetime(claims_bene["dob"], errors="coerce").dt.year
        claims_bene["patient_age"] = (2009 - bene_birth_year).clip(lower=0, upper=115)

        # Sum of chronic illnesses
        chronic_cols = [
            "chronic_alzheimer", "chronic_heartfailure", "chronic_kidneydisease",
            "chronic_cancer", "chronic_copd", "chronic_depression", "chronic_diabetes",
            "chronic_ischemicheart", "chronic_osteoporosis", "chronic_rheumatoidarthritis",
            "chronic_stroke"
        ]
        claims_bene["chronic_count"] = claims_bene[chronic_cols].sum(axis=1)

        bene_agg = claims_bene.groupby("provider_id").agg(
            distinct_beneficiaries=("bene_id", "nunique"),
            mean_patient_age=("patient_age", "mean"),
            renal_disease_ratio=("renal_disease", "mean"),
            mean_chronic_conditions=("chronic_count", "mean"),
            alzheimer_ratio=("chronic_alzheimer", "mean"),
            heartfailure_ratio=("chronic_heartfailure", "mean"),
            kidneydisease_ratio=("chronic_kidneydisease", "mean"),
            diabetes_ratio=("chronic_diabetes", "mean"),
            ischemic_heart_ratio=("chronic_ischemicheart", "mean"),
        ).reset_index()

        # 5. Same-Day Duplicate Billing Detection
        same_day_counts = claims.groupby(["provider_id", "bene_id", "claim_start_date"]).size()
        same_day_duplicates = same_day_counts[same_day_counts > 1].groupby("provider_id").sum().reset_index(name="same_day_duplicate_claims")

        # 6. Merge All Features
        logger.info("Merging feature matrices into unified provider table...")
        df_features = providers.merge(fin_agg, on="provider_id", how="left")
        df_features = df_features.merge(inp_agg, on="provider_id", how="left")
        df_features = df_features.merge(phys_agg, on="provider_id", how="left")
        df_features = df_features.merge(bene_agg, on="provider_id", how="left")
        df_features = df_features.merge(same_day_duplicates, on="provider_id", how="left")

        # Fill missing values for providers with no inpatient claims or same-day duplicates
        df_features["mean_length_of_stay"] = df_features["mean_length_of_stay"].fillna(0.0)
        df_features["max_length_of_stay"] = df_features["max_length_of_stay"].fillna(0.0)
        df_features["distinct_procedures"] = df_features["distinct_procedures"].fillna(0.0)
        df_features["same_day_duplicate_claims"] = df_features["same_day_duplicate_claims"].fillna(0.0)

        # Derived composite ratios
        df_features["claims_per_beneficiary"] = df_features["total_claims"] / (df_features["distinct_beneficiaries"] + 1e-5)
        df_features["physician_to_claim_ratio"] = (
            df_features["distinct_attending_physicians"] / (df_features["total_claims"] + 1e-5)
        )
        df_features["reimbursement_per_beneficiary"] = (
            df_features["total_reimbursed"] / (df_features["distinct_beneficiaries"] + 1e-5)
        )

        logger.info(f"Feature engineering complete. Shape: {df_features.shape}")
        return df_features


if __name__ == "__main__":
    fe = FeatureEngineer()
    df = fe.compute_features()
    print("Features sample:")
    print(df[["provider_id", "potential_fraud", "total_claims", "mean_reimbursed", "claims_per_beneficiary"]].head())
