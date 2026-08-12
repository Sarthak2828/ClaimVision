"""
ClaimVision Data Ingestion Module.
Responsible for acquiring, validating, and caching CMS Medicare Claims datasets.
"""

import os
import logging
import requests
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard CMS Medicare Healthcare Fraud Detection Dataset URLs
DATA_SOURCES = {
    "train_labels": "https://raw.githubusercontent.com/sumeetshahu/Healthcare-Fraud-Detection/main/Data/Original/Training/Train.csv",
    "beneficiary": "https://raw.githubusercontent.com/sumeetshahu/Healthcare-Fraud-Detection/main/Data/Original/Training/Train_Beneficiarydata.csv",
    "inpatient": "https://raw.githubusercontent.com/sumeetshahu/Healthcare-Fraud-Detection/main/Data/Original/Training/Train_Inpatientdata.csv",
    "outpatient": "https://raw.githubusercontent.com/sumeetshahu/Healthcare-Fraud-Detection/main/Data/Original/Training/Train_Outpatientdata.csv",
}

FILE_NAMES = {
    "train_labels": "Train.csv",
    "beneficiary": "Train_Beneficiarydata.csv",
    "inpatient": "Train_Inpatientdata.csv",
    "outpatient": "Train_Outpatientdata.csv",
}


def download_file(url: str, target_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a file with streaming to handle large datasets efficiently."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {target_path.name} from {url}...")
    headers = {"User-Agent": "ClaimVision-ETL/1.0"}
    with requests.get(url, stream=True, headers=headers, timeout=60) as response:
        response.raise_for_status()
        total_bytes = 0
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    total_bytes += len(chunk)
    logger.info(f"Successfully downloaded {target_path.name} ({total_bytes / (1024*1024):.2f} MB)")


def ingest_data(
    raw_dir: Optional[Path] = None,
    force_download: bool = False,
    sample_mode: bool = False,
    sample_provider_count: int = 500,
) -> Dict[str, pd.DataFrame]:
    """
    Ingest CMS Medicare data, downloading if not cached.
    
    Args:
        raw_dir: Path to directory for storing raw CSVs.
        force_download: If True, re-download files even if cached.
        sample_mode: If True, creates a representative sample for rapid execution.
        sample_provider_count: Number of providers to sample if sample_mode is True.
        
    Returns:
        Dictionary containing DataFrames: 'labels', 'beneficiary', 'inpatient', 'outpatient'.
    """
    if raw_dir is None:
        raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download missing files
    for key, filename in FILE_NAMES.items():
        file_path = raw_dir / filename
        if force_download or not file_path.exists() or file_path.stat().st_size == 0:
            download_file(DATA_SOURCES[key], file_path)
        else:
            logger.info(f"Found cached dataset: {file_path.name} ({file_path.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Load datasets into pandas
    logger.info("Loading raw datasets into memory...")
    df_labels = pd.read_csv(raw_dir / FILE_NAMES["train_labels"])
    df_beneficiary = pd.read_csv(raw_dir / FILE_NAMES["beneficiary"])
    df_inpatient = pd.read_csv(raw_dir / FILE_NAMES["inpatient"])
    df_outpatient = pd.read_csv(raw_dir / FILE_NAMES["outpatient"])

    logger.info(f"Loaded records: Labels={len(df_labels)}, Beneficiary={len(df_beneficiary)}, "
                f"Inpatient={len(df_inpatient)}, Outpatient={len(df_outpatient)}")

    # 3. Handle sample mode if requested (preserves stratified fraud ratio)
    if sample_mode and len(df_labels) > sample_provider_count:
        logger.info(f"Sample mode active: filtering down to {sample_provider_count} providers...")
        fraud_ratio = df_labels["PotentialFraud"].value_counts(normalize=True)["Yes"]
        n_fraud = max(1, int(sample_provider_count * fraud_ratio))
        n_non_fraud = sample_provider_count - n_fraud

        fraud_provs = df_labels[df_labels["PotentialFraud"] == "Yes"].sample(n=n_fraud, random_state=42)
        non_fraud_provs = df_labels[df_labels["PotentialFraud"] == "No"].sample(n=n_non_fraud, random_state=42)
        df_labels_sampled = pd.concat([fraud_provs, non_fraud_provs]).reset_index(drop=True)
        sampled_providers = set(df_labels_sampled["Provider"])

        df_inpatient = df_inpatient[df_inpatient["Provider"].isin(sampled_providers)].reset_index(drop=True)
        df_outpatient = df_outpatient[df_outpatient["Provider"].isin(sampled_providers)].reset_index(drop=True)
        active_benes = set(df_inpatient["BeneID"]).union(set(df_outpatient["BeneID"]))
        df_beneficiary = df_beneficiary[df_beneficiary["BeneID"].isin(active_benes)].reset_index(drop=True)
        df_labels = df_labels_sampled

        logger.info(f"Sampled records: Providers={len(df_labels)}, Inpatient={len(df_inpatient)}, "
                    f"Outpatient={len(df_outpatient)}, Active Beneficiaries={len(df_beneficiary)}")

    return {
        "labels": df_labels,
        "beneficiary": df_beneficiary,
        "inpatient": df_inpatient,
        "outpatient": df_outpatient,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest CMS Medicare Claims data.")
    parser.add_argument("--sample", action="store_true", help="Load representative sample")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    data = ingest_data(sample_mode=args.sample, force_download=args.force)
    print("Ingestion complete.")
