# ClaimVision — Power BI Dashboard Implementation Guide

## Executive Overview
The ClaimVision analytical pipeline generates curated analytical datasets and dashboard specifications for Power BI. The solution delivers an executive-ready, decision-support reporting architecture designed to translate machine learning predictions and claims data engineering pipelines into actionable business insights for claims directors, Special Investigation Unit (SIU) analysts, and chief medical officers.

Curated tables are exported to `data/analytical/` by `src/powerbi_exporter.py`, and this guide defines the complete data model, star schema relationships, visual specifications, and layout for loading into Microsoft Power BI Desktop.

---

## Data Model & Relationships

The Power BI data model is structured in a **Star Schema** using the curated datasets in `data/analytical/`:

```
          ┌───────────────────────────┐
          │      kpi_summary          │ (Single Row Portfolio Snapshot)
          └───────────────────────────┘

┌────────────────────────────┐         ┌───────────────────────────────┐
│   provider_risk_scores     │ 1 ─── * │  claims_drillthrough_analytical│
│ (Provider Dimension + ML)  │         │       (Fact Table)            │
└────────────────────────────┘         └───────────────────────────────┘
                                                       *
                                                       │
                                                       │ 1
┌────────────────────────────┐         ┌───────────────────────────────┐
│  state_geographic_summary  │         │       temporal_trends         │
│     (State Dimension)      │         │      (Date / Month Dim)       │
└────────────────────────────┘         └───────────────────────────────┘
```

### Key Relationships
1. `provider_risk_scores[provider_id]` ──(1:Many)──> `claims_drillthrough_analytical[provider_id]`
2. `temporal_trends[claim_year_month]` ──(1:Many)──> `claims_drillthrough_analytical[claim_start_date]` (formatted)
3. `state_geographic_summary[state_id]` ──(1:Many)──> `claims_drillthrough_analytical[state_id]`

---

## 4-Page Dashboard Layout & Visual Specifications

### Page 1: Executive Portfolio Overview
**Audience:** C-Suite, VP of Claims, Chief Risk Officer.
**Objective:** High-level summary of portfolio risk exposure, financial health, and monthly trends.

1. **Top KPI Banner (4 Cards):**
   - **Card 1: Total Claims Volume** (`COUNTROWS(claims_drillthrough_analytical)`)
   - **Card 2: Total Reimbursement Payout** (`$556.54M`) with sub-metric for Average Claim Cost.
   - **Card 3: Portfolio Fraud Rate %** (`9.35%` base provider rate, `38.1%` claim exposure) with RAG indicator.
   - **Card 4: Total Fraud Financial Exposure** (`$212.8M` at risk).
2. **Monthly Claim Payout & Volume Trend (Combo Chart):**
   - **X-axis:** `claim_year_month` (Nov 2008 – Dec 2009).
   - **Column Values:** `total_reimbursement` (bar).
   - **Line Values:** `total_claims` (line).
3. **Inpatient vs Outpatient Financial Mix (Donut Chart):**
   - Inpatient claims: 7.25% of volume driving 73.4% of dollars ($408.3M).
   - Outpatient claims: 92.75% of volume driving 26.6% of dollars ($148.2M).
4. **State Fraud Concentration (Choropleth Map / Bar Chart):**
   - Top fraud exposure states (e.g. State 5, State 10, State 33).

---

### Page 2: Provider Fraud & Risk Deep-Dive
**Audience:** Special Investigation Unit (SIU) Directors, Medical Directors.
**Objective:** Pinpoint anomalous billing behaviors, high-risk provider clusters, and outlier institutions.

1. **Provider Risk Scatter Plot (Quadrant Analysis):**
   - **X-axis:** `total_claims` (Log scale).
   - **Y-axis:** `mean_reimbursed` (Average Payout per Claim).
   - **Bubble Size:** `total_reimbursed`.
   - **Color / Legend:** `risk_tier` (High Risk = Red, Medium Risk = Amber, Low Risk = Green).
   - **Insight:** Highlights high-volume, high-reimbursement outliers in the upper-right quadrant.
2. **Top-20 High Risk Providers Queue (Interactive Grid):**
   - Columns: `provider_id`, `risk_tier`, `fraud_risk_probability`, `total_reimbursed`, `claims_per_beneficiary`, `same_day_duplicate_claims`.
   - Conditional formatting: Heatmap background on `fraud_risk_probability`.
3. **Slicers:**
   - Provider Risk Tier (All / High / Medium / Low).
   - State Filter.
   - Minimum Claim Volume Slider.

---

### Page 3: Clinical & Beneficiary Risk Patterns
**Audience:** Health Informatics, Clinical Policy Team.
**Objective:** Analyze clinical diagnoses, patient chronic comorbidities, and length-of-stay anomalies.

1. **Comorbidity vs Fraud Payout Analysis (Clustered Bar Chart):**
   - Compares patient chronic conditions (Alzheimer's, Heart Failure, Kidney Disease, Diabetes) across Fraudulent vs Legitimate providers.
   - Proves that fraudulent providers disproportionately bill for high-acuity chronic patients.
2. **Inpatient Length-of-Stay Outliers (Histogram / Box Plot):**
   - Shows distribution of inpatient hospital stays, highlighting stays > 15 days.
3. **Same-Day Duplicate Claims Analysis (Bar Chart):**
   - Illustrates duplicate billing frequency per patient on identical service dates.

---

### Page 4: SIU Analyst Claim Investigation Drill-Through
**Audience:** Fraud Case Investigators, Claims Auditors.
**Objective:** Drill down into specific claims and physician networks to build audit evidence.

1. **Provider Header Card:**
   - Dynamic measure: `[Selected Provider Risk Summary]`.
2. **Claim-Level Audit Table:**
   - Columns: `claim_id`, `claim_type`, `claim_start_date`, `reimbursed_amount`, `deductible_amount`, `length_of_stay`, `attending_physician`, `primary_diagnosis_code`.
3. **Attending Physician Network Complexity (Bar Chart):**
   - Distinct physicians billed under the provider to detect physician churn schemes.
4. **Action Workflow:**
   - "Flag for Investigation", "Refer to Audit", "Approve Claim" annotation guide.

---

## How to Load Data into Power BI Desktop

1. Open **Power BI Desktop**.
2. Click **Get Data** -> **Text/CSV**.
3. Navigate to the `data/analytical/` folder in the repository.
4. Import the following CSV files:
   - `kpi_summary.csv`
   - `provider_risk_scores.csv`
   - `temporal_trends.csv`
   - `state_geographic_summary.csv`
   - `claims_drillthrough_analytical.csv`
5. In the **Model View**, verify relationships as documented in the diagram above.
6. Open the **DAX Measures** tab and copy-paste measures from `dax_measures.md`.
7. Build the 4 pages following the visual specifications above.
