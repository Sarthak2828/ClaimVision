"""
ClaimVision Power BI Data Export Layer.
Transforms relational and ML prediction outputs into curated, denormalized,
star-schema analytical tables optimized for Power BI Desktop import and DAX modeling.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parent.parent))
from database.db_manager import DatabaseManager
from src.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PowerBIExporter:
    """Generates analytics-ready export files for Power BI dashboards."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        model_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        self.db = db_manager or DatabaseManager()
        if model_path is None:
            model_path = Path(__file__).resolve().parent.parent / "models" / "xgboost_fraud_model.joblib"
        self.model_path = Path(model_path)
        
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / "data" / "analytical"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self) -> Dict[str, Path]:
        """Generate all 5 analytical dashboard tables."""
        logger.info("Starting Power BI analytical exports...")
        fe = FeatureEngineer(self.db)
        df_features = fe.compute_features()

        # 1. Load trained model for inference
        logger.info(f"Loading trained model from {self.model_path}...")
        artifact = joblib.load(self.model_path)
        model = artifact["model"]
        feature_names = artifact["feature_names"]
        tuned_threshold = artifact.get("best_threshold", 0.5)

        # 2. Score all providers
        logger.info("Computing fraud risk probabilities for all providers...")
        X = df_features[feature_names]
        probs = model.predict_proba(X)[:, 1]
        
        df_provider_scores = df_features.copy()
        df_provider_scores["fraud_risk_probability"] = np.round(probs, 4)
        df_provider_scores["model_fraud_prediction"] = (probs >= tuned_threshold).astype(int)
        
        # Risk Tiers: High, Medium, Low
        conditions = [
            (probs >= tuned_threshold),
            (probs >= 0.30) & (probs < tuned_threshold),
            (probs < 0.30)
        ]
        tier_labels = ["High Risk", "Medium Risk", "Low Risk"]
        df_provider_scores["risk_tier"] = np.select(conditions, tier_labels, default="Low Risk")
        
        # Save Table 1: Provider Risk Scores
        p_scores_path = self.output_dir / "provider_risk_scores.csv"
        df_provider_scores.to_csv(p_scores_path, index=False)
        logger.info(f"Exported Table 1: {p_scores_path.name} ({len(df_provider_scores)} rows)")

        # 3. Save Table 2: Portfolio Executive KPI Summary
        logger.info("Computing Executive KPI Summary...")
        with self.db.engine.connect() as conn:
            kpi_res = conn.execute(text("""
                SELECT 
                    COUNT(*) AS total_claims,
                    COUNT(DISTINCT bene_id) AS distinct_beneficiaries,
                    COUNT(DISTINCT provider_id) AS total_providers,
                    ROUND(SUM(reimbursed_amount), 2) AS total_reimbursement,
                    ROUND(AVG(reimbursed_amount), 2) AS avg_reimbursement_per_claim,
                    ROUND(SUM(deductible_amount), 2) AS total_deductibles
                FROM fact_claims_unified
            """)).mappings().one()

        fraud_prov_count = int(df_features["potential_fraud"].sum())
        fraud_exposure = float(
            df_provider_scores[df_provider_scores["potential_fraud"] == 1]["total_reimbursed"].sum()
        )
        total_payout = float(kpi_res["total_reimbursement"])

        df_kpi = pd.DataFrame([{
            "total_claims": kpi_res["total_claims"],
            "distinct_beneficiaries": kpi_res["distinct_beneficiaries"],
            "total_providers": kpi_res["total_providers"],
            "fraudulent_providers_count": fraud_prov_count,
            "portfolio_fraud_provider_rate_pct": round((fraud_prov_count / len(df_features)) * 100, 2),
            "total_reimbursement_usd": total_payout,
            "total_fraud_loss_exposure_usd": fraud_exposure,
            "fraud_financial_share_pct": round((fraud_exposure / total_payout) * 100, 2),
            "avg_reimbursement_per_claim_usd": kpi_res["avg_reimbursement_per_claim"],
            "total_deductible_usd": kpi_res["total_deductibles"],
        }])
        kpi_path = self.output_dir / "kpi_summary.csv"
        df_kpi.to_csv(kpi_path, index=False)
        logger.info(f"Exported Table 2: {kpi_path.name}")

        # 4. Save Table 3: Temporal Monthly Trends
        logger.info("Computing Temporal Monthly Trends...")
        with self.db.engine.connect() as conn:
            df_trends = pd.read_sql("""
                SELECT 
                    SUBSTR(c.claim_start_date, 1, 7) AS claim_year_month,
                    COUNT(*) AS total_claims,
                    SUM(CASE WHEN c.claim_type = 'Inpatient' THEN 1 ELSE 0 END) AS inpatient_claims,
                    SUM(CASE WHEN c.claim_type = 'Outpatient' THEN 1 ELSE 0 END) AS outpatient_claims,
                    ROUND(SUM(c.reimbursed_amount), 2) AS total_reimbursement,
                    ROUND(AVG(c.reimbursed_amount), 2) AS avg_reimbursement,
                    COUNT(DISTINCT c.provider_id) AS active_providers,
                    COUNT(DISTINCT c.bene_id) AS distinct_beneficiaries
                FROM fact_claims_unified c
                GROUP BY SUBSTR(c.claim_start_date, 1, 7)
                ORDER BY claim_year_month ASC
            """, conn)
        trends_path = self.output_dir / "temporal_trends.csv"
        df_trends.to_csv(trends_path, index=False)
        logger.info(f"Exported Table 3: {trends_path.name} ({len(df_trends)} rows)")

        # 5. Save Table 4: State Geographic Fraud Concentration
        logger.info("Computing State Geographic Fraud Breakdown...")
        with self.db.engine.connect() as conn:
            df_geo = pd.read_sql("""
                SELECT 
                    b.state_id,
                    COUNT(c.claim_id) AS total_claims,
                    COUNT(DISTINCT c.bene_id) AS total_patients,
                    COUNT(DISTINCT c.provider_id) AS total_providers,
                    ROUND(SUM(c.reimbursed_amount), 2) AS total_reimbursement,
                    ROUND(AVG(c.reimbursed_amount), 2) AS avg_reimbursement,
                    SUM(CASE WHEN p.potential_fraud = 1 THEN 1 ELSE 0 END) AS fraud_claims_count,
                    ROUND(100.0 * SUM(CASE WHEN p.potential_fraud = 1 THEN 1 ELSE 0 END) / COUNT(c.claim_id), 2) AS fraud_claim_rate_pct
                FROM fact_claims_unified c
                JOIN dim_beneficiaries b ON c.bene_id = b.bene_id
                JOIN dim_providers p ON c.provider_id = p.provider_id
                GROUP BY b.state_id
                ORDER BY total_claims DESC
            """, conn)
        geo_path = self.output_dir / "state_geographic_summary.csv"
        df_geo.to_csv(geo_path, index=False)
        logger.info(f"Exported Table 4: {geo_path.name} ({len(df_geo)} rows)")

        # 6. Save Table 5: High-Risk Claims Sample for Drill-Through
        logger.info("Computing High-Risk Claims Drill-Through Table...")
        with self.db.engine.connect() as conn:
            df_claims_sample = pd.read_sql("""
                SELECT 
                    c.claim_id,
                    c.claim_type,
                    c.provider_id,
                    p.fraud_label AS ground_truth_fraud,
                    c.bene_id,
                    b.state_id,
                    b.gender,
                    c.claim_start_date,
                    c.claim_end_date,
                    c.reimbursed_amount,
                    c.deductible_amount,
                    c.length_of_stay,
                    c.attending_physician,
                    c.primary_diagnosis_code
                FROM fact_claims_unified c
                JOIN dim_providers p ON c.provider_id = p.provider_id
                JOIN dim_beneficiaries b ON c.bene_id = b.bene_id
                ORDER BY c.reimbursed_amount DESC
                LIMIT 15000
            """, conn)
            
        # Merge model risk tier
        df_claims_sample = df_claims_sample.merge(
            df_provider_scores[["provider_id", "fraud_risk_probability", "risk_tier"]],
            on="provider_id",
            how="left"
        )
        claims_sample_path = self.output_dir / "claims_drillthrough_analytical.csv"
        df_claims_sample.to_csv(claims_sample_path, index=False)
        logger.info(f"Exported Table 5: {claims_sample_path.name} ({len(df_claims_sample)} rows)")

        logger.info("All Power BI analytical exports successfully generated.")
        return {
            "provider_risk_scores": p_scores_path,
            "kpi_summary": kpi_path,
            "temporal_trends": trends_path,
            "state_geographic_summary": geo_path,
            "claims_drillthrough": claims_sample_path,
        }


if __name__ == "__main__":
    exporter = PowerBIExporter()
    exporter.export_all()
