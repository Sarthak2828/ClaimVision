# ClaimVision — Execution Architecture Specification
## Complete Data Flow, Component Breakdown & Step-by-Step Execution Guide

---

## 1. Overall System Architecture Diagram

```
                     ┌─────────────────────────────────────────────────────────────┐
                     │          CMS Medicare Healthcare Fraud Dataset              │
                     │  - Train.csv (5,410 Providers with Fraud Labels)            │
                     │  - Train_Beneficiarydata.csv (138,556 Patient Demographics) │
                     │  - Train_Inpatientdata.csv (40,474 Hospital Claims)         │
                     │  - Train_Outpatientdata.csv (517,737 Clinic Claims)         │
                     └──────────────────────────────┬──────────────────────────────┘
                                                    │
                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: INGESTION & DATA VALIDATION                                                             │
│                                                                                                  │
│  [src/data_ingestion.py]   ──▶ Streams and caches 4 CSV files into data/raw/                     │
│                                (Supports --sample mode for rapid 15s developer testing)          │
│                                                                                                  │
│  [src/data_validator.py]   ──▶ Verifies schema, asserts range boundaries, audits target leakage   │
│                                (Confirms 0 forbidden columns, valid dates, 9.35% fraud rate)     │
└───────────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: 3NF STAR-SCHEMA ETL & RELATIONAL LOADING                                                │
│                                                                                                  │
│  [src/etl_pipeline.py]     ──▶ Transforms raw files into normalized entities:                    │
│                                - dim_providers (5,410 rows)                                      │
│                                - dim_beneficiaries (138,556 rows)                                │
│                                - fact_inpatient_claims (40,474 rows)                             │
│                                - fact_outpatient_claims (517,737 rows)                           │
│                                - fact_claims_unified (558,211 rows)                              │
│                                                                                                  │
│  [database/db_manager.py]  ──▶ Executes DDL (schema.sql) and loads SQLite via SQLAlchemy        │
│                                (Standard bulk executemany inserts; post-load count assertions)   │
└───────────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: ADVANCED SQL ANALYTICS & BENCHMARKING                                                   │
│                                                                                                  │
│  [database/run_queries.py] ──▶ Executes 18 production SQL queries across 5 modules:              │
│                                - aggregations.sql (Executive KPIs, Inpatient vs Outpatient)      │
│                                - joins_subqueries.sql (Multi-CTE patient chronic profiles)       │
│                                - window_functions.sql (DENSE_RANK within state, NTILE deciles)   │
│                                - fraud_patterns.sql (Phantom billing, length-of-stay outliers)   │
│                                - query_optimization.sql (EXPLAIN covering index scan: 0.36 ms)   │
└───────────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: LEAKAGE-FREE FEATURE ENGINEERING                                                        │
│                                                                                                  │
│  [src/feature_engineering.py] ──▶ Extracts 33 behavioral provider features from analytical DB:   │
│                                  - Financial: Total/Mean/Std/Max payout, deductible ratios       │
│                                  - Clinical: Inpatient LOS, procedure code counts                │
│                                  - Network: Distinct attending & operating physicians, churn     │
│                                  - Demographics: Patient chronic illness ratios (ESRD, Heart)    │
│                                  - Behavioral: Same-day duplicate patient billing count          │
│                                  - Output: 5,410 x 35 feature matrix (zero target leakage)       │
└───────────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: MACHINE LEARNING & THRESHOLD OPTIMIZATION                                               │
│                                                                                                  │
│  [src/model_pipeline.py]   ──▶ 1. Stratified 80/20 train/test split (Train=4,328, Test=1,082)   │
│                                2. Benchmark baselines: Dummy (AUC=0.50), LogisticReg (AUC=0.969) │
│                                3. Class Imbalance: scale_pos_weight = <dynamic>                       │
│                                4. 5-Fold Stratified Cross-Validation on XGBoost (Mean AUC=0.9448)│
│                                5. Threshold Optimization: Calibrates threshold = 0.7674          │
│                                6. Holdout Test Evaluation:                                       │
│                                   - ROC-AUC: 0.9557                                              │
│                                   - PR-AUC: 0.7369                                               │
│                                   - Tuned Precision: 66.00%                                      │
│                                   - Tuned Recall: 65.35%                                         │
│                                   - Tuned F1-Score: 0.6567                                       │
│                                7. Serializes models/xgboost_fraud_model.joblib                   │
└───────────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: EXPLAINABLE AI (SHAP) ATTRIBUTIONS                                                      │
│                                                                                                  │
│  [src/explainability.py]   ──▶ 1. TreeExplainer attribution with XGBoost 3.x UBJSON patch        │
│                                2. Global Beeswarm plot -> reports/figures/shap_summary_beeswarm  │
│                                3. Feature Importance bar -> reports/figures/shap_feature_import. │
│                                4. Local Waterfall plot -> reports/figures/shap_waterfall_PRV...  │
│                                5. Generates structured plain-English case briefs for SIU audit   │
└───────────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 7: BUSINESS ANALYTICS & POWER BI EXPORT LAYER                                              │
│                                                                                                  │
│  [src/powerbi_exporter.py] ──▶ Generates 5 curated Star-Schema CSVs in data/analytical/:        │
│                                1. provider_risk_scores.csv (5,410 rows with probability & tiers) │
│                                2. kpi_summary.csv (Executive portfolio snapshot)                │
│                                3. temporal_trends.csv (Monthly claims and payout trends)         │
│                                4. state_geographic_summary.csv (State fraud concentrations)      │
│                                5. claims_drillthrough_analytical.csv (15,000 drill-down rows)    │
│                                                                                                  │
│  [powerbi/dax_measures.md] ──▶ 25+ DAX Measures (Base, Rates, RAG formatting, Time-Intel)       │
│  [powerbi/POWER_BI_GUIDE.md]──▶ 4-Page Dashboard Layout & Visual Blueprint                       │
└───────────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 8: AUTOMATED TESTING & VERIFICATION                                                        │
│                                                                                                  │
│  [pytest -v]               ──▶ 12 automated unit tests across validation, ETL, features, ML,    │
│                                and SQL contracts (100% pass rate in 14.28s)                      │
│                                                                                                  │
│  [run_pipeline.py]         ──▶ End-to-end master pipeline orchestrator                           │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Repository Structure & Artifact Locations

```
ClaimVision/
├── configs/
│   └── config.yaml                     # Model hyperparameters, dataset paths, thresholds
├── data/
│   ├── raw/                            # Downloaded CMS Medicare CSVs (gitignored)
│   │   ├── Train.csv
│   │   ├── Train_Beneficiarydata.csv
│   │   ├── Train_Inpatientdata.csv
│   │   └── Train_Outpatientdata.csv
│   ├── processed/
│   │   └── claimvision.db              # SQLite 3NF Star-Schema Database (gitignored)
│   └── analytical/                     # Power BI Export CSVs (gitignored)
│       ├── provider_risk_scores.csv
│       ├── kpi_summary.csv
│       ├── temporal_trends.csv
│       ├── state_geographic_summary.csv
│       └── claims_drillthrough_analytical.csv
├── database/
│   ├── schema.sql                      # DDL for dimensions, facts & indexes
│   ├── db_manager.py                   # SQLAlchemy engine manager
│   ├── run_queries.py                  # SQL execution runner & profiler
│   └── queries/
│       ├── aggregations.sql            # Q1-Q5: Business KPIs & summaries
│       ├── joins_subqueries.sql        # Q6-Q9: Multi-table joins & CTEs
│       ├── window_functions.sql        # Q10-Q13: DENSE_RANK, NTILE, Moving Averages
│       ├── fraud_patterns.sql          # Q14-Q17: Phantom billing & length-of-stay outliers
│       └── query_optimization.sql      # Q18: EXPLAIN QUERY PLAN benchmarking
├── models/
│   └── xgboost_fraud_model.joblib      # Serialized trained model & metadata (gitignored)
├── powerbi/
│   ├── dax_measures.md                 # Complete DAX formula library
│   └── POWER_BI_GUIDE.md               # 4-page visual layout and relationship guide
├── reports/
│   ├── model_metrics.json              # Real evaluation metrics JSON
│   ├── shap_feature_importance.json    # SHAP global feature rankings
│   └── figures/
│       ├── shap_feature_importance.png # Mean |SHAP| bar chart
│       ├── shap_summary_beeswarm.png   # Beeswarm distribution plot
│       └── shap_waterfall_PRV54742.png # Local provider case explanation
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py               # Streaming dataset downloader
│   ├── data_validator.py               # Schema, range & leakage audit
│   ├── etl_pipeline.py                 # Star schema ETL & database loader
│   ├── feature_engineering.py          # 33 provider behavioral features
│   ├── model_pipeline.py               # XGBoost, class weighting, CV, threshold tuning
│   ├── explainability.py               # SHAP TreeExplainer & narrative generator
│   └── powerbi_exporter.py             # Export curated tables for Power BI
├── tests/
│   ├── __init__.py
│   ├── test_data_validation.py         # Schema & leakage unit tests
│   ├── test_etl_and_database.py        # Transformation & DB loader tests
│   ├── test_feature_engineering.py     # Feature calculation tests
│   ├── test_model_pipeline.py          # XGBoost inference & threshold tests
│   └── test_sql_queries.py             # SQL syntax & execution tests
├── .env.example                        # Database and environment template
├── .gitignore                          # Exclusions for large data, cache & models
├── requirements.txt                    # Pinned Python dependencies
├── run_pipeline.py                     # Master execution runner
├── SKILL.md                            # Comprehensive technical skills reference
├── EXECUTION_ARCHITECTURE.md           # This execution architecture specification
└── README.md                           # Executive summary and documentation
```

---

## 3. Step-by-Step Execution Order

### Prerequisites
1. Python 3.10+ installed.
2. Clone repository:
   ```bash
   git clone https://github.com/Sarthak2828/ClaimVision.git
   cd ClaimVision
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution Steps
To execute the entire system from raw data ingestion to Power BI exports with a single command:
```bash
python run_pipeline.py
```

Or execute individual stages modularly:

#### Step 1: Ingest & Validate Raw Data
```bash
python src/data_ingestion.py
python src/data_validator.py
```
- Downloads and caches CMS Medicare files into `data/raw/`.
- Executes data contracts, range validations, and confirms zero target leakage.

#### Step 2: Run Star-Schema ETL & Database Loading
```bash
python src/etl_pipeline.py
```
- Creates tables and indexes in `data/processed/claimvision.db`.
- Loads 5,410 providers, 138,556 beneficiaries, and 558,211 claims.

#### Step 3: Execute SQL Analytics & Performance Profiling
```bash
python database/run_queries.py
```
- Runs 18 analytical queries across aggregations, CTEs, window functions, and fraud heuristics.
- Verifies covering index optimization via `EXPLAIN QUERY PLAN`.

#### Step 4: Extract Features & Train XGBoost Classifier
```bash
python src/feature_engineering.py
python src/model_pipeline.py
```
- Aggregates 33 provider features.
- Trains Dummy, Logistic Regression, and XGBoost with `scale_pos_weight=<dynamic>`.
- Tunes decision threshold to 0.7674.
- Evaluates holdout test set (ROC-AUC: 0.9557, F1: 0.6567, Precision: 66.00%, Recall: 65.35%).
- Serializes model to `models/xgboost_fraud_model.joblib`.

#### Step 5: Generate SHAP Attributions & Visual Explanations
```bash
python src/explainability.py
```
- Computes TreeExplainer attributions.
- Generates beeswarm summary and local waterfall plots in `reports/figures/`.
- Outputs natural language case brief for SIU analysts.

#### Step 6: Export Analytical Tables for Power BI
```bash
python src/powerbi_exporter.py
```
- Generates 5 curated CSV files in `data/analytical/`.

#### Step 7: Run Automated Test Suite
```bash
pytest -v
```
- Runs 12 automated unit and integration tests (100% pass rate).

---

## 4. Definition of Done
The project is considered complete and production-ready when:
1. All 558,211 claims and 5,410 providers are successfully loaded and relational integrity verified.
2. All 18 SQL queries execute without syntax errors and demonstrate query tuning.
3. Feature engineering executes with strictly zero target leakage.
4. Model achieves ROC-AUC > 0.95 and F1 > 0.70 with deliberate threshold tuning.
5. SHAP visualizations and case briefs are generated in `reports/figures/`.
6. Power BI tables and DAX measures library are fully documented and loadable.
7. All 12 pytest tests pass cleanly.
8. All code is committed and successfully pushed to `https://github.com/Sarthak2828/ClaimVision.git`.
