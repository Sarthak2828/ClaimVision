# ClaimVision - Healthcare Claims Intelligence Platform

> **ClaimVision is a healthcare claims intelligence pipeline that uses Python, SQL and PostgreSQL for data validation, ETL, feature engineering and provider-level fraud-risk classification. XGBoost is used for classification, SHAP provides model explainability, and analytical outputs are prepared for Power BI-based business analysis.**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16.15-336791.svg)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker_Compose-PostgreSQL_16-2496ED.svg)](https://www.docker.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44%2B-brightgreen.svg)](https://shap.readthedocs.io/)
[![Power BI](https://img.shields.io/badge/Power_BI-Curated_Datasets-yellow.svg)](https://powerbi.microsoft.com/)
[![Tests](https://img.shields.io/badge/tests-18%20passed-success.svg)](https://pytest.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Project Overview

ClaimVision is designed to address healthcare reimbursement fraud by identifying high-risk healthcare providers through behavioral, financial, and clinical claim patterns.

In Medicare billing, fraud labels exist at the **provider level** (`PotentialFraud`), not at the level of individual claims. ClaimVision aggregates and transforms claims and beneficiary histories into provider-level behavioral profiles, evaluates risk using a leakage-safe provider-group split, explains predictions with SHAP, and exports curated analytical tables for Power BI reporting.

### End-to-End Architecture

```
Raw CMS healthcare claims data
        |
Data validation
        |
ETL / transformation
        |
PostgreSQL
        |
SQL analytics
        |
Provider-level feature engineering
        |
Provider-group train/test split
        |
XGBoost classification
        |
Evaluation
        |
SHAP explainability
        |
Power BI-ready analytical outputs
```

---

## Dataset Scale & Entity Breakdown

All statistics are derived from the Centers for Medicare & Medicaid Services (CMS) DE-SynPUF synthetic claims dataset:

| Entity / Dimension | Volume / Value | Description |
|---|---|---|
| **Healthcare Providers** | **5,410** | Distinct hospitals, clinics, and physician practices |
| **Total Claims Analyzed** | **558,211** | Unified claims portfolio (2008-2009 service period) |
| Inpatient Claims | 40,474 | Hospital admissions with length of stay, procedures, and diagnoses |
| Outpatient Claims | 517,737 | Ambulatory and clinical facility visits |
| **Covered Beneficiaries** | **138,556** | Distinct Medicare beneficiaries with demographic & chronic profiles |
| **Total Reimbursements Billed** | **$556,543,140.00** | Total claims payout across the entire portfolio |
| **Fraudulent Providers (Ground Truth)** | **506** (9.35%) | Providers with confirmed investigation flags (`PotentialFraud = Yes`) |
| **Fraud Financial Exposure** | **$212,796,000.00** | Payout volume associated with flagged fraudulent providers (38.2%) |

---

## Evaluation Methodology & Model Results

### Provider-Level Group Split (Leakage Prevention)
Because multiple claims originate from the same healthcare provider, standard row-level train/test splitting causes severe data leakage: provider-specific billing practices appear in both training and test sets.

ClaimVision enforces a **provider-level group split**:
- Every provider is assigned **entirely** to the training set or **entirely** to the test set.
- Class distribution is preserved across splits (~9.35% fraud rate).
- Provider disjoint constraint is strictly enforced: `set(train_ids).isdisjoint(set(test_ids))` gives **0 provider overlap**.

| Split Partition | Provider Count | Fraud Providers | Clean Providers | Fraud Rate |
|---|---:|---:|---:|---:|
| **Training Set** | 4,329 | 405 | 3,924 | 9.36% |
| **Holdout Test Set** | 1,081 | 101 | 980 | 9.34% |
| **Total** | 5,410 | 506 | 4,904 | 9.35% |

### Dynamic Class Weighting
To handle the ~9.35% class imbalance, `scale_pos_weight` is calculated **dynamically from training data only**:

```
scale_pos_weight = N_negative_train / N_positive_train = 3,924 / 405 = 9.6889
```

No test set label information is used in weighting or model fitting.

### Authoritative Model Results

Evaluated on the 1,081 holdout test providers:

| Metric | Provider-Level Holdout |
|---|---:|
| **ROC-AUC** | **0.9557** |
| **PR-AUC** | **0.7369** |
| **F1-Score** | **0.6567** |
| **Precision** | **66.00%** |
| **Recall** | **65.35%** |
| **Decision Threshold** | **0.7674** |
| **5-Fold CV ROC-AUC** | **0.9477** |

#### Holdout Confusion Matrix (at Tuned Threshold 0.7674)
- **True Negatives (TN)**: 946
- **False Positives (FP)**: 34
- **False Negatives (FN)**: 35
- **True Positives (TP)**: 66

---

## Technology Stack

| Component | Technology | Role in Platform |
|---|---|---|
| **Core Runtime** | Python 3.10+ | Orchestration, ETL, validation, and modeling |
| **Database Engine** | PostgreSQL 16 (via Docker Compose) | Canonical relational store with star schema and B-Tree indexes |
| **Database Driver & ORM** | Psycopg2-binary, SQLAlchemy 2.0+ | Connection pooling and high-throughput PostgreSQL COPY streaming |
| **Data Processing** | Pandas 2.0+, NumPy | Tabular transformations and provider feature engineering |
| **Machine Learning** | XGBoost 2.0+, Scikit-Learn 1.3+ | Gradient boosted trees, threshold tuning, and stratified cross-validation |
| **Explainability** | SHAP 0.44+ | TreeExplainer global feature importance and local provider attributions |
| **Business Intelligence** | Power BI Desktop, DAX | Analytical dataset consumption, data modeling, and KPI reporting |
| **Quality & Testing** | Pytest 7.4+ | 18 automated unit and integration tests |

---

## Relational Database & Star Schema

The canonical database is **PostgreSQL 16** managed via `docker-compose.yml`.

### Schema Architecture
The schema models claims and clinical data in a relational star structure:
- **`dim_providers`**: Provider master entity (`provider_id`, `potential_fraud`).
- **`dim_beneficiaries`**: Patient demographics, coverage dates, and 11 chronic condition indicators.
- **`fact_inpatient_claims`**: Inpatient admissions, length of stay, admission/discharge diagnoses, procedure codes.
- **`fact_outpatient_claims`**: Outpatient visits, service dates, reimbursement and deductible amounts.
- **`fact_claims_unified`**: Unified claims view joining inpatient and outpatient events for aggregate querying.

### High-Throughput Bulk Loading
ETL data loading utilizes PostgreSQL's native `COPY FROM STDIN` via `psql_insert_copy`:
- Bulk loads all 558,211 claims and dimension tables in approximately 15 seconds.
- Enforces relational foreign key constraints and domain check constraints.

### SQL Query Suite & Plan Inspection
The repository includes 18 analytical SQL queries across 5 modules (`database/queries/`):
1. **Aggregations & KPIs**: Total claims, financial payouts, outpatient vs. inpatient ratios.
2. **Joins & Multi-CTE Profiles**: Patient chronic condition comorbidity patterns by provider risk.
3. **Window Functions**: `DENSE_RANK` state-level payout hierarchies and running cumulative totals.
4. **Fraud Heuristics**: Same-day duplicate billing detection and physician network complexity ratios.
5. **Query Plan Inspection**: PostgreSQL `EXPLAIN (ANALYZE, BUFFERS, COSTS)` inspection demonstrating index scans and hash aggregation performance.

---

## Model Explainability (SHAP)

ClaimVision implements **SHAP (SHapley Additive exPlanations)** using `TreeExplainer` to interpret how engineered features contribute to the model's provider fraud-risk predictions.

**Note on Interpretation:** SHAP feature attributions reflect feature contributions to the tree ensemble's log-odds output; they describe statistical association within the model, not medical or legal causation.

### Key Global Risk Drivers
- **`total_reimbursed`**: High aggregate financial volume strongly increases predicted risk.
- **`max_length_of_stay`**: Disproportionately long inpatient stays contribute positively to risk scores.
- **`total_claims`** & **`inpatient_claims`**: Excessive billing volume relative to peer medians.
- **`physician_to_claim_ratio`**: High physician churn or abnormally high claims per physician.

### Generated Artifacts
- Global summary beeswarm plot: `reports/figures/shap_summary_beeswarm.png`
- Global feature importance bar chart: `reports/figures/shap_feature_importance.png`
- Local waterfall explanation briefs: `reports/figures/shap_waterfall_PRV54742.png`

---

## Power BI Analytical Layer

ClaimVision generates 5 curated, star-schema analytical tables in `data/analytical/` designed for direct import into **Microsoft Power BI Desktop**:

| Table | File | Grain / Content |
|---|---|---|
| **Provider Risk Scores** | `provider_risk_scores.csv` | 5,410 rows - 35 engineered features, risk probability, and assigned risk tier |
| **Executive KPI Summary** | `kpi_summary.csv` | Portfolio-level totals, fraud rates, and model performance metrics |
| **Temporal Monthly Trends** | `temporal_trends.csv` | Monthly claim counts, reimbursements, and volume progression |
| **State Geographic Breakdown** | `state_geographic_summary.csv` | State-level claim volumes, total reimbursements, and fraud concentration |
| **Claims Drilldown Table** | `claims_drillthrough_analytical.csv` | 15,000 claim audit records with attending physician and diagnosis detail |

Complete data model documentation, star-schema relationship specifications, and 4-page dashboard wireframes are provided in [powerbi/POWER_BI_GUIDE.md](powerbi/POWER_BI_GUIDE.md). A library of 25+ DAX calculations is provided in [powerbi/dax_measures.md](powerbi/dax_measures.md).

---

## Repository Structure

```
ClaimVision/
|-- docker-compose.yml                  # Canonical PostgreSQL 16 service definition
|-- .env.example                        # Environment template (DATABASE_URL)
|-- requirements.txt                    # Project Python dependencies
|-- run_pipeline.py                     # Single-command pipeline orchestrator
|-- database/
|   |-- schema.sql                      # PostgreSQL DDL for dimension and fact tables
|   |-- db_manager.py                   # SQLAlchemy connection manager
|   |-- run_queries.py                  # Production SQL analytics runner
|   +-- queries/                        # 18 PostgreSQL analytical queries
|       |-- aggregations.sql
|       |-- joins_subqueries.sql
|       |-- window_functions.sql
|       |-- fraud_patterns.sql
|       +-- query_optimization.sql
|-- data/
|   |-- raw/                            # Ingested CMS Medicare CSVs
|   +-- analytical/                     # 5 curated Power BI export tables
|-- models/
|   +-- xgboost_fraud_model.joblib      # Serialized trained model, scaler & split metadata
|-- powerbi/
|   |-- POWER_BI_GUIDE.md               # Power BI data model & 4-page dashboard guide
|   +-- dax_measures.md                 # 25+ DAX formulas library
|-- reports/
|   |-- model_metrics.json              # Authoritative model evaluation results
|   |-- shap_feature_importance.json    # Global SHAP feature ranking data
|   +-- figures/                        # Generated SHAP visualization plots
|-- src/
|   |-- data_ingestion.py               # Dataset ingestion & local caching
|   |-- data_validator.py               # Schema integrity & target leakage audits
|   |-- etl_pipeline.py                 # Star schema ETL & PostgreSQL COPY loader
|   |-- feature_engineering.py          # 33 provider behavioral feature signatures
|   |-- model_pipeline.py               # Provider group split, CV, XGBoost training
|   |-- explainability.py               # SHAP TreeExplainer & waterfall narratives
|   +-- powerbi_exporter.py             # Curated analytical table generation
+-- tests/                              # Automated pytest suite (18 tests)
    |-- test_data_validation.py
    |-- test_etl_and_database.py
    |-- test_feature_engineering.py
    |-- test_model_pipeline.py
    +-- test_sql_queries.py
```

---

## Quick Start & Reproducibility

### 1. Clone the Repository
```bash
git clone https://github.com/Sarthak2828/ClaimVision.git
cd ClaimVision
```

### 2. Configure Environment & Start PostgreSQL
Copy the environment template and start the canonical PostgreSQL container:
```bash
cp .env.example .env
docker compose up -d
```
Verify the container is healthy:
```bash
docker compose ps
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Test Suite
Execute all 18 unit, integration, and database tests:
```bash
pytest -v
```
All 18 tests will execute and pass against the running PostgreSQL service.

### 5. Run the Pipeline
End-to-end pipeline: Data Ingestion, Validation, ETL, PostgreSQL Loading, SQL Analytics, Feature Engineering, XGBoost Training, SHAP, Power BI Exports:

```bash
# Full dataset execution (~558k claims, ~1 minute):
python run_pipeline.py

# Rapid developer sample mode (500 providers, ~15 seconds):
python run_pipeline.py --sample
```

---

## License & Attribution
- **Dataset**: Centers for Medicare & Medicaid Services (CMS) 2008-2010 Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF). Public domain.
- **Inspiration**: Problem formulation and domain reference inspired by `gagan8605/ClaimVision`. Re-architected with PostgreSQL 16, provider-group splitting, dynamic class weighting, and leakage-safe evaluation.
- **License**: Released under the [MIT License](LICENSE).
