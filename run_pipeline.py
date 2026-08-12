"""
ClaimVision Master Pipeline Orchestrator.
Executes the end-to-end healthcare claims fraud analytics pipeline with a single command:
Ingestion -> Validation -> ETL -> SQL Analytics -> Feature Engineering -> ML -> SHAP -> Power BI
"""

import sys
import time
import argparse
import logging
from pathlib import Path

# Add src and database to path
sys.path.append(str(Path(__file__).resolve().parent))
from src.data_ingestion import ingest_data
from src.data_validator import DataValidator
from src.etl_pipeline import ClaimsETLPipeline
from src.feature_engineering import FeatureEngineer
from src.model_pipeline import FraudModelPipeline
from src.explainability import FraudExplainability
from src.powerbi_exporter import PowerBIExporter
from database.run_queries import run_all_queries

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ClaimVision-Master")


def execute_pipeline(sample_mode: bool = False, skip_etl: bool = False) -> None:
    """Execute complete end-to-end ClaimVision platform workflow."""
    start_total = time.perf_counter()
    logger.info("=" * 80)
    logger.info("CLAIMVISION — HEALTHCARE CLAIMS INTELLIGENCE PLATFORM")
    logger.info(f"Execution Mode: {'SAMPLE (Fast Mode)' if sample_mode else 'FULL DATASET'}")
    logger.info("=" * 80)

    # 1. Ingestion & Validation & ETL
    if not skip_etl:
        logger.info("\n>>> STAGE 1: Data Ingestion & Star-Schema ETL...")
        etl = ClaimsETLPipeline()
        etl.run(sample_mode=sample_mode, sample_provider_count=500)
    else:
        logger.info("\n>>> Skipping Stage 1 ETL (Database already populated).")

    # 2. SQL Analytics
    logger.info("\n>>> STAGE 2: Executing Production SQL Analytics Suite...")
    run_all_queries()

    # 3. Feature Engineering
    logger.info("\n>>> STAGE 3: Leakage-Free Feature Engineering...")
    fe = FeatureEngineer()
    df_features = fe.compute_features()

    # 4. Machine Learning Classification
    logger.info("\n>>> STAGE 4: XGBoost Model Training, CV & Threshold Selection...")
    ml = FraudModelPipeline()
    metrics = ml.run(df_features)

    # 5. Model Explainability (SHAP)
    logger.info("\n>>> STAGE 5: Explainable AI (SHAP) Attribution Generation...")
    xai = FraudExplainability()
    xai.generate_explanations(df_features)

    # 6. Power BI Analytical Export
    logger.info("\n>>> STAGE 6: Power BI Analytical Table Generation...")
    pbi = PowerBIExporter()
    pbi.export_all()

    elapsed = time.perf_counter() - start_total
    logger.info("=" * 80)
    logger.info(f"CLAIMVISION PIPELINE EXECUTION COMPLETE in {elapsed:.2f} seconds!")
    logger.info(f"Final Model Test ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Final Model Tuned F1-Score: {metrics['tuned_threshold']['f1_score']:.4f}")
    logger.info(f"Final Model Tuned Precision: {metrics['tuned_threshold']['precision']:.4f}")
    logger.info(f"Final Model Tuned Recall: {metrics['tuned_threshold']['recall']:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClaimVision Master Pipeline")
    parser.add_argument("--sample", action="store_true", help="Execute rapid sample mode")
    parser.add_argument("--skip-etl", action="store_true", help="Skip ETL if database is already populated")
    args = parser.parse_args()

    execute_pipeline(sample_mode=args.sample, skip_etl=args.skip_etl)
