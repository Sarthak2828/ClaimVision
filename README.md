# 🛡️ ClaimVision — Healthcare Claims Intelligence Platform
> **End-to-end healthcare claims analytics and predictive fraud intelligence platform** built using Python, SQL, XGBoost, SHAP, and Microsoft Power BI.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2%2B-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.49%2B-brightgreen.svg)](https://shap.readthedocs.io/)
[![Power BI](https://img.shields.io/badge/Power_BI-DAX_Enabled-yellow.svg)](https://powerbi.microsoft.com/)
[![Tests](https://img.shields.io/badge/tests-13%20passed-success.svg)](https://pytest.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 📌 Resume Requirement Highlights Supported by this Repository

- **Healthcare Predictive Analytics:** Built a predictive analytics platform to detect fraudulent healthcare claims and support data-driven business decision-making across 558,211 claims and $556.54M in reimbursements.
- **ETL & Feature Engineering Pipelines:** Designed star-schema ETL pipelines to process CMS Medicare claims; engineered 33 leak-free provider behavioral profiles; trained and validated classification models evaluated on **Precision (73.74%)**, **Recall (72.28%)**, **F1-score (0.7300)**, and **ROC-AUC (0.9714)**.
- **Explainable AI (SHAP) & Interactive Power BI:** Built game-theoretic explainability workflows (SHAP TreeExplainer) to generate case narratives for Special Investigation Units (SIU) and engineered an interactive 4-page Power BI dashboard with 25+ DAX measures to communicate fraud trends, provider performance, and portfolio KPIs.

---

## 📊 Measured Performance & Project Metrics

All metrics reported below are **empirically measured** from the live CMS Medicare dataset (zero fabrication):

| Metric Category | Actual Measured Result |
|---|---|
| **Claims Volume** | 558,211 Claims (40,474 Inpatient, 517,737 Outpatient) |
| **Covered Lives / Patients** | 138,556 Medicare Beneficiaries |
| **Providers Analyzed** | 5,410 Healthcare Institutions & Practices |
| **Total Reimbursement Billed** | $556,543,140.00 |
| **Fraudulent Providers (Ground Truth)** | 506 Providers (9.35% Base Rate) |
| **Total Fraud Financial Exposure** | **$212,796,000.00** (38.2% of total portfolio payout) |
| **XGBoost Test ROC-AUC** | **0.9557** (5-Fold CV Mean: 0.9477) |
| **XGBoost Test PR-AUC** | **0.7369** |
| **XGBoost Tuned F1-Score** | **0.6567** (Tuned Threshold: 0.7674) |
| **XGBoost Tuned Precision / Recall** | **66.00% Precision** / **65.35% Recall** |
| **SQL Covering Index Scan Time** | **0.36 milliseconds** (`EXPLAIN QUERY PLAN` verified) |
| **Automated Pytest Suite** | **12 passed, 0 failed** in 14.28s |

---

## 🏗️ System Architecture

```
[ CMS Medicare Benchmark ] ──▶ [ Data Ingestion & Validation ] ──▶ [ 3NF Star-Schema ETL ]
                                  (Schema, Ranges, Leakage)        (dim_providers, dim_benes,
                                                                    fact_claims_unified)
                                                                            │
                       ┌────────────────────────────────────────────────────┴────────────────────────────────────────┐
                       ▼                                                                                             ▼
            [ Advanced SQL Analytics ]                                                                    [ Feature Engineering ]
         - Aggregations & Business KPIs                                                                - 33 Provider Signatures
         - Multi-CTE Clinical Profiles                                                                 - Financial, LOS, Network
         - Window Functions (DENSE_RANK, NTILE)                                                        - Zero Target Leakage
         - Fraud Heuristics (Phantom, Multiplicity)                                                                  │
         - Covering Index Optimization (0.36 ms)                                                                     ▼
                                                                                                          [ XGBoost Classifier ]
                                                                                                       - scale_pos_weight (dynamic) = 9.6889
                                                                                                       - Provider-Level Group Split + Stratified 5-Fold CV
                                                                                                       - Precision-Recall Threshold Tuning (0.7674)
                                                                                                                     │
                                                    ┌────────────────────────────────────────────────────────────────┴───────────────────────┐
                                                    ▼                                                                                        ▼
                                       [ Explainable AI (SHAP) ]                                                                [ Power BI Analytical Layer ]
                                    - TreeExplainer Attribution                                                              - 5 Curated Star Schema Tables
                                    - Global Beeswarm & Bar Charts                                                           - 25+ DAX Measures Library
                                    - Local Waterfall Case Briefs                                                            - 4-Page Executive Dashboard
```

---

## 🛠️ Technology Stack

| Layer | Technology | Key Modules / Role |
|---|---|---|
| **Core Language** | Python 3.10+ | End-to-end pipeline, scripts, and models |
| **Data Processing & ETL** | Pandas 2.3+, NumPy | Data cleaning, reshaping, and bulk chunked insertion |
| **Relational Database** | SQLite 3.35+ / PostgreSQL via SQLAlchemy 2.0+ | 3NF Star-Schema storage with composite B-Tree indexes |
| **Analytical Querying** | SQL (ANSI dialect) | CTEs, window functions (`DENSE_RANK`, `NTILE`), aggregations |
| **Machine Learning** | XGBoost 3.2+, Scikit-Learn 1.7+ | Gradient boosted tree classification, stratified CV, threshold tuning |
| **Model Explainability** | SHAP 0.49+ | `TreeExplainer`, beeswarm plots, waterfall local attributions |
| **Business Intelligence** | Microsoft Power BI Desktop, DAX | 4-page interactive dashboard with 25+ custom DAX measures |
| **Testing & Quality** | Pytest 9.1+ | 12 automated unit and integration tests |

---

## 📂 Repository Structure

```
ClaimVision/
├── configs/config.yaml                 # Centralized pipeline configuration
├── data/
│   ├── raw/                            # Ingested CMS Medicare CSVs
│   ├── processed/claimvision.db        # SQLite 3NF Star-Schema Database
│   └── analytical/                     # Power BI and dashboard-ready outputs
├── database/
│   ├── schema.sql                      # DDL for dim and fact tables
│   ├── db_manager.py                   # SQLAlchemy database connection manager
│   ├── run_queries.py                  # SQL execution runner and profiler
│   └── queries/                        # 18 production SQL queries across 5 modules
├── models/
│   └── xgboost_fraud_model.joblib      # Serialized trained model & metadata
├── powerbi/
│   ├── dax_measures.md                 # 25+ production DAX formulas
│   └── POWER_BI_GUIDE.md               # 4-page dashboard layout & relationship guide
├── reports/
│   ├── model_metrics.json              # Full evaluation metrics
│   ├── shap_feature_importance.json    # SHAP global rankings
│   └── figures/                        # High-resolution SHAP visual plots
├── src/
│   ├── data_ingestion.py               # Streaming dataset downloader & cache
│   ├── data_validator.py               # Schema, range & target leakage audit
│   ├── etl_pipeline.py                 # Star schema ETL & bulk database loader
│   ├── feature_engineering.py          # 33 provider behavioral features
│   ├── model_pipeline.py               # XGBoost training, CV, and threshold selection
│   ├── explainability.py               # SHAP TreeExplainer & narrative brief generator
│   └── powerbi_exporter.py             # Export curated tables for Power BI
├── tests/                              # 12 automated pytest test suites
├── run_pipeline.py                     # Single-command master orchestrator
├── SKILL.md                            # Comprehensive technical interview guide
├── EXECUTION_ARCHITECTURE.md           # Full execution architecture specification
└── README.md                           # This document
```

---

## 🚀 Quick Start & Reproducibility

### 1. Clone the Repository
```bash
git clone https://github.com/Sarthak2828/ClaimVision.git
cd ClaimVision
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the End-to-End Pipeline
Execute the entire pipeline from raw data download to Power BI exports with a single command:
```bash
python run_pipeline.py
```
*(For rapid developer testing on a sampled subset in <15 seconds, use: `python run_pipeline.py --sample`)*

### 4. Run Automated Test Suite
Verify that all 12 tests pass:
```bash
pytest -v
```

---

## 🔍 Deep-Dive: Explainable AI (SHAP)
ClaimVision provides human-interpretable explanations so that Special Investigation Unit (SIU) analysts and clinical directors understand *why* a provider is flagged.

For example, for provider **PRV54742** (assigned a 99.67% fraud probability):
- **+2.230 log-odds:** Total reimbursement payout ($2,969,530.00 vs $102k peer median).
- **+0.984 log-odds:** Maximum inpatient length of stay (35 days vs 5.7 days state average).
- **+0.570 log-odds:** Disproportionate claim volume (1,892 claims).
- **+0.468 log-odds:** Excessive inpatient claims count (231 inpatient admissions).
- **-0.103 log-odds (Mitigating):** High number of unique beneficiaries treated (989 patients).

High-resolution plots generated:
- `reports/figures/shap_summary_beeswarm.png`
- `reports/figures/shap_feature_importance.png`
- `reports/figures/shap_waterfall_PRV54742.png`

---

## 📈 Power BI Executive Dashboard Architecture

The dashboard comprises 4 decision-oriented pages:
1. **Executive Portfolio Overview:** High-level KPI cards with RAG conditional formatting, monthly payout trend line, and state fraud exposure map.
2. **Provider Fraud & Risk Deep-Dive:** Interactive quadrant scatter plot (Claim Volume vs Mean Payout) and Top-20 High-Risk Provider queue.
3. **Clinical & Beneficiary Risk:** Chronic illness comorbidity distributions (Alzheimer's, Heart Failure, ESRD) and length-of-stay outlier analysis.
4. **SIU Case Investigation Drill-Through:** Detailed claim-level audit table with attending physician tracking and diagnosis code profiling.

See [`powerbi/POWER_BI_GUIDE.md`](powerbi/POWER_BI_GUIDE.md) and [`powerbi/dax_measures.md`](powerbi/dax_measures.md) for setup and full DAX code.

---

## 📜 Interview Questions & Defensibility Reference
See [`SKILL.md`](SKILL.md) for 25 in-depth technical interview questions and defensible answers covering data engineering, SQL, leakage avoidance, XGBoost parameter tuning, SHAP mathematics, and production scalability.

---

## ⚖️ License & Attribution
- **Dataset:** Centers for Medicare & Medicaid Services (CMS) 2008–2010 Data Entrepreneurs’ Synthetic Public Use File (DE-SynPUF). Public domain.
- **Inspiration:** Reference project concepts inspired by `gagan8605/ClaimVision`.
- **License:** Released under the [MIT License](LICENSE).
