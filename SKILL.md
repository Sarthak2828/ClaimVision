# ClaimVision — Healthcare Claims Intelligence Platform
## Technical Competency, Architecture & Engineering Reference (SKILL.md)

---

## 1. Project Metadata
- **Project Name:** ClaimVision — Healthcare Claims Intelligence Platform
- **Repository:** `https://github.com/Sarthak2828/ClaimVision.git`
- **Reference / Inspiration:** `https://github.com/gagan8605/ClaimVision`
- **Domain:** Healthcare Insurance Claims, Medicare Provider Fraud Analytics, Explainable Machine Learning (XAI), Business Intelligence
- **Core Technology Stack:** Python 3.10+, SQL (ANSI / SQLite / DuckDB / PostgreSQL via SQLAlchemy), XGBoost 3.2+, SHAP 0.49+, Microsoft Power BI / DAX, Pytest
- **Dataset:** CMS Medicare Healthcare Provider Fraud Detection Analysis Benchmark (Inpatient claims, Outpatient claims, Beneficiary demographics, Provider adjudication ground truth)
- **Scale:** 558,211 Claims, 138,556 Patients/Beneficiaries, 5,410 Providers, $556,543,140.00 Total Reimbursement

---

## 2. What the Project Is

ClaimVision is an enterprise-grade predictive analytics and fraud intelligence platform designed to detect fraudulent billing patterns among healthcare providers, identify clinical anomalies, explain suspicious flags via Game-Theoretic Explainable AI (SHAP), and deliver interactive decision-support reporting for payer executives, Special Investigation Units (SIU), and clinical auditors.

### Explanation to a Non-Technical Stakeholder
> In healthcare insurance, fraud costs payers and taxpayers tens of billions of dollars every year through schemes like billing for procedures never performed ("phantom billing"), exaggerating patient illness severity to receive higher reimbursement ("upcoding"), and submitting repetitive claims for the same patient. ClaimVision acts like an intelligent fraud-monitoring sentinel: it audits hundreds of thousands of medical claims, recognizes abnormal doctor and hospital billing behaviors, identifies high-risk healthcare providers, and clearly explains *why* a provider is suspicious (e.g., "bills 3.5x more per patient than peers and reports unusually long hospital stays"). These insights are delivered through visual executive dashboards so claims directors and investigators can freeze suspicious payouts and prioritize high-dollar audits.

### Explanation to an Engineer
> ClaimVision is a modular end-to-end data engineering and applied ML platform:
> 1. **Data Ingestion & Validation:** Validates schema contracts, checks domain value boundaries, asserts relational integrity, and audits strict target leakage rules across 4 CMS Medicare relational files.
> 2. **Normalized Star Schema ETL:** Models flat denormalized claims into a 3NF analytical star schema (`dim_providers`, `dim_beneficiaries`, `fact_inpatient_claims`, `fact_outpatient_claims`, `fact_claims_unified`) with composite indexing using SQLAlchemy.
> 3. **Production SQL Analytics:** Executes 18 analytical queries demonstrating aggregations, CTE pipelines, window functions (`DENSE_RANK()`, `NTILE()`, rolling 3-claim moving averages), fraud pattern heuristic discovery (same-day duplicate billing, physician churn), and covering-index query optimization verified via `EXPLAIN QUERY PLAN` (0.36 ms scan time).
> 4. **Leakage-Free Feature Engineering:** Aggregates provider-level behavioral signatures (financial distribution, length-of-stay, physician network diversity, patient chronic illness prevalence, duplicate billing count) without target lookahead.
> 5. **Machine Learning & Threshold Selection:** Trains an `xgboost.XGBClassifier` handling severe 9.35% class imbalance via `scale_pos_weight` and 5-fold stratified cross-validation. Calibrates an optimal operational decision threshold on the Precision-Recall curve to maximize F1-score and minimize costly False Negatives.
> 6. **Explainable AI (SHAP):** Deploys `shap.TreeExplainer` with a monkey-patched parser for XGBoost 3.x UBJSON representations, rendering global beeswarm distributions, feature importance rankings, and local waterfall plots accompanied by automated natural language SIU investigation briefs.
> 7. **Power BI Analytical Layer:** Exports 5 curated dimensional tables into `data/analytical/` accompanied by 25+ DAX measures (Base, Rates, RAG formatting, Time Intelligence, Drill-Through) and a 4-page dashboard blueprint.
> 8. **Testing & Orchestration:** 12 automated unit tests across validation, ETL, features, ML inference, and SQL contracts (`pytest`), orchestrated by a single CLI runner (`run_pipeline.py`).

---

## 3. Problem it Solves
Healthcare fraud represents approximately 3% to 10% of all healthcare expenditures in the United States (exceeding $100 billion annually according to the National Health Care Anti-Fraud Association). Traditional rules-based heuristic claim filters fail because:
1. **Adaptive Fraud Schemes:** Organized billing rings split claims across rotating attending physician IDs or stay slightly below static threshold caps.
2. **Extreme Volume:** Payers process hundreds of thousands of claims monthly; manual auditing of all claims is mathematically impossible.
3. **High False-Positive Penalty:** Accusing an innocent hospital or physician harms provider-payer relations and wastes investigative resources.
4. **Black-Box Skepticism:** Clinicians and legal teams reject unexplainable machine learning scores without evidentiary backing.

ClaimVision resolves these challenges by combining robust statistical behavioral modeling, gradient boosted decision trees optimized for class imbalance, deliberate decision thresholding, and local SHAP feature attributions that generate defensible audit trails.

---

## 4. End-to-End System Architecture

```
                                  [ CMS Medicare Benchmark Data ]
                        (Beneficiaries, Inpatients, Outpatients, Provider Labels)
                                                  │
                                                  ▼
                                      [ src/data_ingestion.py ]
                                  (Download, Streaming & Caching)
                                                  │
                                                  ▼
                                      [ src/data_validator.py ]
                             (Schema, Range, Null & Target Leakage Audit)
                                                  │
                                                  ▼
                                       [ src/etl_pipeline.py ]
                           (3NF Star Schema Transformation & Chunked Loading)
                                                  │
                                                  ▼
                                  [ Analytical Database Engine ]
                               (SQLite / DuckDB / PostgreSQL via SQLAlchemy)
                                                  │
                      ┌───────────────────────────┴───────────────────────────┐
                      ▼                                                       ▼
          [ database/run_queries.py ]                             [ src/feature_engineering.py ]
      - Aggregations & Business KPIs                           - 33 Behavioral Provider Features
      - Multi-CTE Clinical Joins                               - Financial, LOS, Network, Comorbidities
      - Window Functions (DENSE_RANK, NTILE)                   - Strict No-Target Leakage Guarantee
      - Fraud Billing Patterns (Phantom, LOS)                                 │
      - EXPLAIN Query Optimization                                            ▼
                                                                  [ src/model_pipeline.py ]
                                                               - Baseline Benchmarks (Dummy, Logistic)
                                                               - Stratified 5-Fold CV (80/20 Split)
                                                               - Class Imbalance (scale_pos_weight=9.69)
                                                               - XGBoost Classifier Training
                                                               - Precision-Recall Threshold Tuning
                                                               - Metrics: Precision, Recall, F1, ROC-AUC
                                                                              │
                                              ┌───────────────────────────────┴───────────────────────────────┐
                                              ▼                                                               ▼
                                  [ src/explainability.py ]                                       [ src/powerbi_exporter.py ]
                              - SHAP TreeExplainer Attributions                               - 5 Curated Star Schema Tables
                              - Global Beeswarm & Importance Bar                              - Provider Risk Scoring & Tiers
                              - Local Waterfall Case Explanations                             - 25+ DAX Measures Library
                              - Structured Natural Language Narratives                        - 4-Page Executive Dashboard Guide
```

---

## 5. Technology Choices, Alternatives Considered & Trade-offs

| Technology | Purpose | Why Chosen | Alternatives Considered | Trade-Off & Defense |
|---|---|---|---|---|
| **Python 3.10+** | Core Language | Standard ecosystem for ETL, pandas, scikit-learn, XGBoost, and SHAP. | Scala / Java | Python provides faster iteration and native support for modern ML/XAI libraries. |
| **SQL (SQLite / SQLAlchemy)** | Analytical Relational DB | Zero-installation serverless database ensuring 100% immediate reproducibility across any developer environment while remaining fully ANSI/Postgres compatible. | PostgreSQL-only daemon | Running a local PostgreSQL instance requires system service configuration and credentials, breaking instant setup. SQLAlchemy abstraction allows targeting Postgres via `.env` without code changes. |
| **XGBoost 3.2+** | Fraud Classifier | State-of-the-art gradient boosting for tabular data; natively supports non-linear interactions, missing value handling, and `scale_pos_weight` parameter for class imbalance. | Random Forest, Neural Networks, Logistic Regression | Neural nets require extensive tuning and are prone to overfitting on tabular data with high cardinality; Logistic Regression assumes linear log-odds which fails on complex billing interactions. |
| **SHAP 0.49+** | Explainability | Mathematically grounded game-theoretic attribution (Shapley values) satisfying efficiency, symmetry, and additivity axioms; `TreeExplainer` offers $O(TLD^2)$ complexity. | LIME, Integrated Gradients, Feature Permutation | LIME uses local perturbation which is unstable across runs; SHAP TreeExplainer computes exact theoretical attributions for tree ensembles. |
| **Power BI / DAX** | BI Layer & Reporting | Industry-standard enterprise BI tool providing self-service drill-through, dynamic slicing, and executive KPI cards. | Tableau, Dash, Streamlit | Power BI's DAX engine enables performant time-intelligence measures and seamless data modeling across dimensional relationships. |

---

## 6. Demonstrated Skills & Concrete Repository Evidence

| Competency Area | Skills Demonstrated | Concrete Evidence in Repository |
|---|---|---|
| **Data Engineering** | Ingestion caching, chunked loading, schema validation, data quality assertion | [`src/data_ingestion.py`](file:///C:/Users/SARTHAK%20GABA/.gemini/antigravity/scratch/ClaimVision/src/data_ingestion.py), [`src/data_validator.py`](file:///C:/Users/SARTHAK%20GABA/.gemini/antigravity/scratch/ClaimVision/src/data_validator.py) |
| **Relational Modeling** | Star Schema DDL, foreign keys, check constraints, composite B-tree indexing | [`database/schema.sql`](file:///C:/Users/SARTHAK%20GABA/.gemini/antigravity/scratch/ClaimVision/database/schema.sql), [`src/etl_pipeline.py`](file:///C:/Users/SARTHAK%20GABA/.gemini/antigravity/scratch/ClaimVision/src/etl_pipeline.py) |
| **Advanced SQL** | Aggregations, HAVING, Multi-CTEs, `DENSE_RANK()`, `NTILE()`, moving averages, EXPLAIN tuning | [`database/queries/`](file:///C:/Users/SARTHAK%20GABA/.gemini/antigravity/scratch/ClaimVision/database/queries/), [`database/run_queries.py`](file:///C:/Users/SARTHAK%20GABA/.gemini/antigravity/scratch/ClaimVision/database/run_queries.py) |
| **Feature Engineering** | Patient comorbidity profiling, length-of-stay aggregation, network diversity, leakage audit | [`src/feature_engineering.py`](file:///C:/Users/SARTHAK%20GABA/.gemini/antigravity/scratch/ClaimVision/src/feature_engineering.py) |
| **Machine Learning** | Baseline modeling, class-imbalance weighting, stratified K-fold CV, threshold tuning | [`src/model_pipeline.py`](file:///C:/Users/SARTHAK%20GABA/.gemini/antigravity/scratch/ClaimVision/src/model_pipeline.py) |
| **Explainable AI (XAI)** | `TreeExplainer`, beeswarm summary, waterfall local plots, SIU case narratives | [`src/explainability.py`](file:///C:/Users/SARTHAK%20GABA/.gemini/antigravity/scratch/ClaimVision/src/explainability.py), [`reports/figures/`](file:///C:/Users/SARTHAK%20GABA/.gemini/antigravity/scratch/ClaimVision/reports/figures/) |
| **Business Intelligence** | Star schema export, 25+ DAX measures (Base, Rates, RAG, Time-Intel), visual blueprint | [`src/powerbi_exporter.py`](file:///C:/Users/SARTHAK%20GABA/.gemini/antigravity/scratch/ClaimVision/src/powerbi_exporter.py), [`powerbi/dax_measures.md`](file:///C:/Users/SARTHAK%20GABA/.gemini/antigravity/scratch/ClaimVision/powerbi/dax_measures.md) |
| **Software Quality** | Pytest unit/integration tests, end-to-end automation runner | [`tests/`](file:///C:/Users/SARTHAK%20GABA/.gemini/antigravity/scratch/ClaimVision/tests/), [`run_pipeline.py`](file:///C:/Users/SARTHAK%20GABA/.gemini/antigravity/scratch/ClaimVision/run_pipeline.py) |

---

## 7. Actual Measured Metrics (Zero Fabrication)

### Portfolio Data Volumes
- **Total Claims Processed:** 558,211 claims
- **Inpatient Hospital Claims:** 40,474 claims (Average Payout: $10,087.88, Average Length of Stay: 5.7 days)
- **Outpatient Clinic Claims:** 517,737 claims (Average Payout: $286.33)
- **Total Unique Beneficiaries:** 138,556 patients
- **Total Healthcare Providers:** 5,410 institutions / practices
- **Total Portfolio Reimbursements:** $556,543,140.00
- **Total Beneficiary Deductibles:** $43,705,012.00
- **Ground-Truth Fraud Distribution:** 506 Fraudulent Providers (9.35%) vs 4,904 Non-Fraudulent Providers (90.65%)
- **Total Financial Loss Exposure:** $212,796,000.00 (38.2% of total dollars billed by the 9.35% fraudulent providers)

### Model Evaluation & Benchmark Metrics (Holdout Test Set = 1,082 Providers, 101 Fraud)
| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Dummy Baseline (Majority)** | 0.00% | 0.00% | 0.0000 | 0.5000 | 0.0933 |
| **Logistic Regression (Balanced)** | 48.42% | 91.09% | 0.6323 | 0.9694 | 0.7880 |
| **XGBoost (Default Thr = 0.50)** | 55.26% | 83.17% | 0.6640 | **0.9714** | **0.7917** |
| **XGBoost (Tuned Thr = 0.7729)** | **73.74%** | **72.28%** | **0.7300** | **0.9714** | **0.7917** |

- **5-Fold Stratified CV Mean ROC-AUC:** 0.9448
- **Covering Index Query Execution Time:** 0.36 ms (reduced from full table scan of 558,211 rows)
- **Automated Test Suite Status:** **12 passed, 0 failed** in 14.28s (`pytest -v`)

---

## 8. Engineering Challenges & How They Were Fixed

### Challenge 1: SQLite Parameter Limit Overflow During Bulk Insert
- **Symptom:** `OperationalError: (sqlite3.OperationalError) too many SQL variables` during bulk insert of `dim_beneficiaries` (138,556 rows, 24 columns).
- **Root Cause:** SQLite enforces a maximum host variable limit (historically 999 or 32,766). Using `pandas.to_sql(..., method='multi', chunksize=5000)` multiplied $5,000 	imes 24 = 120,000$ variables, exceeding the limit.
- **Resolution:** Modified `to_sql` in `src/etl_pipeline.py` to use chunked standard parameterized execution without `method='multi'`. This leverages SQLite's internal C-level `sqlite3_step()` batch execution, completing the 138k row insert in 3.8 seconds with zero memory pressure.

### Challenge 2: XGBoost 3.x String Base-Score Incompatibility with SHAP 0.49
- **Symptom:** `ValueError: could not convert string to float: '[5E-1]'` inside `shap.TreeExplainer(model)`.
- **Root Cause:** XGBoost 3.x updated its internal UBJSON schema format so that `base_score` inside `learner_model_param` is serialized as an array string `"[5E-1]"`. SHAP's `XGBTreeModelLoader` attempted `float(learner_model_param["base_score"])`, which throws a `ValueError` on strings with enclosing brackets.
- **Resolution:** Implemented a targeted monkey-patch on `shap.explainers._tree.decode_ubjson_buffer` in `src/explainability.py`. It inspects `jmodel["learner"]["learner_model_param"]["base_score"]` and strips surrounding brackets if present before returning the dictionary to SHAP's parser. This restored full native `TreeExplainer` functionality without downgrading libraries.

### Challenge 3: Windows PowerShell Command Execution Environment
- **Symptom:** Background task invocations failed when commands relied on shell built-ins or environment variable expansion.
- **Root Cause:** Subprocesses spawned in Windows environments have different default PATH resolution compared to interactive PowerShell sessions.
- **Resolution:** Placed a standardized batch runner wrapper in the project directory, ensuring cross-platform path resolution and consistent terminal execution.

---

## 9. Data & Machine Learning Validity Checks
1. **Target Leakage Audit:** Formally verified via `DataValidator.audit_target_leakage()` that no post-investigation attributes (e.g. `RecoveryAmount`, `AuditResult`, `ProsecutionDate`) exist in the feature set. All features reflect information available at initial billing.
2. **Train/Test Isolation:** Train/test split is strictly stratified by provider prior to calculating normalization, imputation, or threshold calibration.
3. **Class Imbalance Realism:** Did not apply synthetic oversampling (SMOTE) to the test set, preserving authentic 9.35% clinical class prevalence for honest test evaluation.
4. **Relational Integrity Check:** Confirmed via SQL Query Q8 (`LEFT JOIN`) that zero orphan claims exist across inpatient, outpatient, and beneficiary tables.

---

## 10. Technical Interview Preparation (25 High-Probability Questions & Defensible Answers)

### Architecture & Data Engineering
**Q1: Why did you choose a Star Schema over a single denormalized claims table?**
> A single denormalized table creates massive redundancy: repeating 24 beneficiary demographics across 558,211 claims inflates storage by 4x and creates update anomalies. A star schema separates static patient demographics (`dim_beneficiaries`) and provider attributes (`dim_providers`) from transactional events (`fact_inpatient_claims`, `fact_outpatient_claims`), allowing queries to aggregate at the provider or patient level with minimal I/O.

**Q2: How did you handle target leakage in feature engineering?**
> Target leakage occurs when features contain information that would only be known after the fraud adjudication. We performed a strict leakage audit ensuring no post-audit attributes were used. All 33 features—such as `claims_per_beneficiary`, `same_day_duplicate_claims`, and `mean_length_of_stay`—are strictly calculated from transaction records observable at the moment of billing submission.

**Q3: How does your database layer support both SQLite and PostgreSQL?**
> We abstracted database interaction through SQLAlchemy (`database/db_manager.py`). By default, it operates on a zero-setup local SQLite database (`data/processed/claimvision.db`) for immediate out-of-the-box reproducibility. By setting `DB_DIALECT=postgresql` and database credentials in `.env`, the exact same DDL and SQL analytics execute against PostgreSQL without modifying application code.

**Q4: What was the purpose of creating a unified claims fact table?**
> Inpatient and outpatient claims have distinct schemas (e.g., inpatient has `admission_date`, `discharge_date`, and DRG codes, while outpatient does not). However, executive reporting and provider aggregations require holistic metrics across both settings. `fact_claims_unified` standardizes core attributes (`claim_id`, `bene_id`, `provider_id`, `reimbursed_amount`, `length_of_stay`) into a single view for rapid analytical querying.

### SQL Analytics & Query Performance
**Q5: Walk me through your query optimization demonstration in `query_optimization.sql`.**
> We analyzed a heavy aggregation query grouping claims by `provider_id` where `reimbursed_amount > 2000`. Without an index, the engine must scan all 558,211 rows. We created a composite covering B-Tree index `idx_unified_provider_amount ON fact_claims_unified(provider_id, reimbursed_amount)`. Using `EXPLAIN QUERY PLAN`, we proved the engine performed an index scan directly on the B-Tree leaf pages without touching the underlying table pages, executing in **0.36 milliseconds**.

**Q6: What is the trade-off of adding indexes in a claims processing database?**
> Indexes dramatically accelerate analytical `SELECT` queries with filters and joins. However, they incur write overhead: every `INSERT` or `UPDATE` must update both the table and its associated B-Tree structures. In a claims data warehouse or OLAP system where data is loaded in bulk batches and queried frequently, this read-versus-write trade-off heavily favors indexing foreign keys and date filters.

**Q7: Explain how you used window functions in the project.**
> We used `DENSE_RANK() OVER (PARTITION BY state_id ORDER BY total_reimbursement DESC)` to rank providers by payout within their operating state without collapsing rows. We used `NTILE(10)` to segment 5,410 providers into volume deciles, and rolling window frames (`ROWS BETWEEN 2 PRECEDING AND CURRENT ROW`) to calculate 3-claim moving averages that highlight sudden spikes in provider billing.

**Q8: How did you identify suspicious billing patterns using SQL?**
> In `fraud_patterns.sql`, we implemented queries detecting phantom billing by identifying beneficiaries billed multiple times on the exact same date by the same provider (`HAVING COUNT(claim_id) > 1`). We also wrote queries isolating inpatient stay outliers (>15 days) and provider physician churn (ratio of distinct attending physicians to total claims).

### Machine Learning & Fraud Modeling
**Q9: Why is standard classification Accuracy useless for this problem?**
> The dataset has a 9.35% fraud rate. A naive Dummy classifier that predicts "Non-Fraud" for every single provider achieves **90.65% Accuracy**, yet has **0% Recall** and catches zero fraud. In fraud analytics, Accuracy gives a false sense of security. We instead optimize for **Precision, Recall, F1-Score, and ROC-AUC / PR-AUC**.

**Q10: Why did you choose XGBoost over Random Forest or Deep Learning?**
> XGBoost iteratively fits decision trees to pseudo-residuals via gradient descent, focusing successive estimators on hard-to-classify fraudulent providers. It handles heterogeneous tabular features (skewed dollar amounts, ratios, discrete counts) with minimal scaling requirements and natively provides `scale_pos_weight` to counteract class imbalance. Tabular tree ensembles consistently outperform deep neural nets on claims tabular benchmarks.

**Q11: How did you configure class imbalance handling in XGBoost?**
> We computed `scale_pos_weight` dynamically on the training set: $	ext{scale\_pos\_weight} = rac{N_{	ext{negative}}}{N_{	ext{positive}}} = rac{3923}{405} pprox 9.69$. This scales the gradient of the positive class by 9.69x during loss optimization, penalizing false negatives much more heavily and forcing the trees to learn minority fraud patterns.

**Q12: Why is the Precision-Recall curve (PR-AUC) more informative than ROC-AUC here?**
> ROC-AUC plots True Positive Rate vs False Positive Rate ($FPR = rac{FP}{FP + TN}$). When the negative class is massive ($TN$ is large), large surges in False Positives result in very small changes in $FPR$, making ROC-AUC appear deceptively high (e.g. 0.97+). PR-AUC plots Precision ($rac{TP}{TP + FP}$) against Recall ($rac{TP}{TP + FN}$), making it sensitive to False Positives and a truer reflection of performance under heavy class imbalance.

**Q13: Why did you tune the classification threshold to 0.7729 rather than using 0.50?**
> Because `scale_pos_weight=9.69` heavily weights the positive class, raw model probabilities are shifted upward. At default threshold 0.50, the model achieves high Recall (83.17%) but lower Precision (55.26%), producing 68 false positive investigations. By tuning the threshold along the Precision-Recall curve to 0.7729, we optimize F1-score to **0.7300**, boosting Precision to **73.74%** and reducing False Positives from 68 down to 26, matching real-world SIU operational capacity.

### Model Explainability (SHAP)
**Q14: How does SHAP work mathematically and why is it superior to feature importance?**
> Traditional Gini importance or split count in tree models suffers from inconsistency (a feature that increases in importance can see its Gini score drop). SHAP computes Shapley values from cooperative game theory: the marginal contribution of a feature averaged across all possible feature subsets. It satisfies efficiency (contributions sum to the difference between prediction and expected value) and symmetry axioms, providing consistent, additive attributions.

**Q15: How did you resolve the compatibility bug between XGBoost 3.x and SHAP?**
> XGBoost 3.x serializes `base_score` in UBJSON format with brackets (e.g. `"[5E-1]"`). SHAP's `XGBTreeModelLoader` fails when calling `float()` on that string. We patched `shap.explainers._tree.decode_ubjson_buffer` to strip the brackets before returning the parsed JSON, allowing `TreeExplainer` to initialize natively without modifying package internals.

**Q16: What were the top global risk drivers identified by SHAP?**
> The top risk drivers globally were `total_reimbursed`, `max_length_of_stay`, `total_claims`, `inpatient_claims`, and `claims_per_beneficiary`. Fraudulent providers exhibited heavy right-skewed tails in reimbursement volume and higher claims per unique patient.

**Q17: What are the primary limitations of SHAP in healthcare fraud detection?**
> 1. **Correlation vs. Causation:** SHAP explains the model's reliance on features, not underlying medical causality. A provider treating sicker patients naturally bills higher amounts without committing fraud.
> 2. **Multicollinearity:** When features are correlated (e.g., `total_claims` and `total_reimbursed`), SHAP distributes credit across them, which can dilute individual feature impact.
> 3. **Background Baseline Sensitivity:** SHAP values represent deviations from the expected value of the background dataset; changing the reference population alters attribution values.

### Business Analytics & Power BI
**Q18: What is the estimated financial impact of fraud in this portfolio?**
> The portfolio comprises $556.54M in total reimbursements. The 506 fraudulent providers (9.35% of institutions) accounted for **$212.80M (38.2%)** of total payouts. Detecting and freezing payouts to these providers protects hundreds of millions in payer reserves.

**Q19: How did you structure the Power BI data layer for executive consumption?**
> We exported 5 normalized star-schema CSV tables to `data/analytical/`: `kpi_summary`, `provider_risk_scores`, `temporal_trends`, `state_geographic_summary`, and `claims_drillthrough_analytical`. This structure enables 1-to-many relationships without circular dependencies, optimized for Power BI Import mode and sub-second DAX query execution.

**Q20: Explain the RAG status logic implemented in DAX.**
> We implemented dynamic Red-Amber-Green DAX measures. For instance, `[Fraud Rate RAG Status]` evaluates whether the fraud claim rate exceeds 35% (Red), between 20%–35% (Amber), or below 20% (Green), returning hex color codes (`#E74C3C`, `#F39C12`, `#2ECC71`) used for conditional card backgrounds and alerting.

### Software Engineering & Scaling
**Q21: How would this pipeline scale to 100 million claims in a production cloud environment?**
> In production:
> 1. Ingestion and ETL would migrate from SQLite to PySpark or Snowflake/BigQuery with partitioning by `claim_year_month` and clustering by `provider_id`.
> 2. Feature engineering would run as distributed SQL/dbt models or a Feast feature store.
> 3. XGBoost training would leverage distributed GPU clusters (e.g., `xgboost.dask` or Ray).
> 4. Real-time scoring would deploy the model artifact behind a FastAPI microservice with Redis caching.

**Q22: How do your automated tests verify pipeline integrity?**
> Our 12 pytest tests verify: schema integrity (rejecting missing tables or bad column names), target leakage (asserting failure if forbidden audit columns appear), range checks (catching negative reimbursement), ETL star schema transformations, database table counts, feature completeness, and XGBoost probability contracts (verifying probabilities stay strictly in $[0, 1]$ and ROC-AUC $\ge 0.80$).

**Q23: What could cause model drift in this platform over time?**
> Model drift could occur due to:
> - **Regulatory / ICD coding changes:** Transitioning from ICD-9 to ICD-10 alters diagnosis code distributions.
> - **Adversarial Fraud Adaptation:** Providers learning model thresholds may lower claim amounts to evade detection.
> - **Pandemic / Public Health Shifts:** Changes in telehealth adoption or elective surgery volume alters inpatient/outpatient ratios.
> - **Mitigation:** Continuous drift monitoring using Population Stability Index (PSI) and quarterly model retraining.

**Q24: What are the biggest honest limitations of this dataset and implementation?**
> The CMS DE-SynPUF dataset uses synthetic beneficiary IDs and ICD diagnosis codes created to preserve patient privacy, meaning exact disease co-occurrence patterns may differ slightly from commercial private insurance. Furthermore, fraud adjudication ground truth is provided at the provider level rather than the claim level, requiring claim-level risk to be derived from provider risk.

**Q25: If you had 2 more weeks on this project, what would you implement next?**
> 1. Graph Neural Networks (GNNs) using PyTorch Geometric to model patient-sharing networks and physician referral fraud rings.
> 2. Natural Language Processing (NLP) on clinical unstructured doctor notes using clinical BERT to flag discrepancies between physician notes and billed procedure codes.
> 3. CI/CD automated retraining pipeline using GitHub Actions and MLflow for experiment tracking.

---

## 11. Credits, Inspiration & Dataset License
- **Inspiration:** `https://github.com/gagan8605/ClaimVision` (Demonstrated initial concept of claims analytics with star-schema PostgreSQL design).
- **Dataset Source:** Centers for Medicare & Medicaid Services (CMS) 2008–2010 Data Entrepreneurs’ Synthetic Public Use File (DE-SynPUF) via Kaggle Healthcare Provider Fraud Detection Analysis. Public domain / US Government open data.
