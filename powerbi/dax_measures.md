# ClaimVision — Power BI DAX Measures Library

This document provides the production-grade DAX measures library used in ClaimVision. The measures are organized across 5 functional groups: Base Volume & Financials, Rates & Risk Ratios, RAG Conditional Formatting, Time Intelligence, and Drill-Through Context.

---

## 1. Base Measures (Financial & Volume)

### `[Total Claims]`
Calculates the aggregate count of all healthcare claims in the portfolio.
```dax
Total Claims = COUNTROWS('claims_drillthrough_analytical')
```

### `[Total Reimbursement]`
Calculates the total payout amount reimbursed by Medicare/payer.
```dax
Total Reimbursement = SUM('claims_drillthrough_analytical'[reimbursed_amount])
```

### `[Avg Claim Cost]`
Average financial payout per claim.
```dax
Avg Claim Cost = AVERAGE('claims_drillthrough_analytical'[reimbursed_amount])
```

### `[Total Deductibles Paid]`
Total deductible amount paid by beneficiaries.
```dax
Total Deductibles Paid = SUM('claims_drillthrough_analytical'[deductible_amount])
```

### `[Active Providers]`
Count of distinct healthcare providers currently active.
```dax
Active Providers = DISTINCTCOUNT('provider_risk_scores'[provider_id])
```

### `[Total Beneficiaries]`
Count of distinct insured Medicare beneficiaries.
```dax
Total Beneficiaries = DISTINCTCOUNT('claims_drillthrough_analytical'[bene_id])
```

---

## 2. Fraud & Risk Ratio Measures

### `[Fraud Claims Count]`
Count of claims filed by providers with ground-truth or confirmed fraud classification.
```dax
Fraud Claims Count = 
CALCULATE(
    COUNTROWS('claims_drillthrough_analytical'),
    'claims_drillthrough_analytical'[ground_truth_fraud] = "Yes"
)
```

### `[Portfolio Fraud Rate %]`
Proportion of total claims flagged as fraudulent.
```dax
Portfolio Fraud Rate % = 
DIVIDE(
    [Fraud Claims Count],
    [Total Claims],
    0
) * 100
```

### `[Predicted High-Risk Providers]`
Count of providers assigned to the "High Risk" tier by the XGBoost classification model.
```dax
Predicted High-Risk Providers = 
CALCULATE(
    COUNTROWS('provider_risk_scores'),
    'provider_risk_scores'[risk_tier] = "High Risk"
)
```

### `[High Risk Provider %]`
Percentage of active providers identified as High Risk.
```dax
High Risk Provider % = 
DIVIDE(
    [Predicted High-Risk Providers],
    [Active Providers],
    0
) * 100
```

### `[Total Fraud Loss Exposure]`
Total dollar volume reimbursed to fraudulent providers.
```dax
Total Fraud Loss Exposure = 
CALCULATE(
    SUM('provider_risk_scores'[total_reimbursed]),
    'provider_risk_scores'[potential_fraud] = 1
)
```

### `[Inpatient Cost Ratio %]`
Share of total reimbursements attributable to inpatient hospital stays.
```dax
Inpatient Cost Ratio % = 
VAR InpatientCost = 
    CALCULATE(
        [Total Reimbursement],
        'claims_drillthrough_analytical'[claim_type] = "Inpatient"
    )
RETURN
    DIVIDE(InpatientCost, [Total Reimbursement], 0) * 100
```

---

## 3. RAG Status & Conditional Formatting Measures

### `[Fraud Rate RAG Status]`
Assigns Red-Amber-Green status based on portfolio fraud benchmarks.
- Green: < 20%
- Amber: 20% - 35%
- Red: > 35%
```dax
Fraud Rate RAG Status = 
SWITCH(
    TRUE(),
    [Portfolio Fraud Rate %] > 35, "RED",
    [Portfolio Fraud Rate %] >= 20, "AMBER",
    "GREEN"
)
```

### `[Fraud Rate Hex Color]`
Generates dynamic hex color codes for KPI card backgrounds and visual alerts.
```dax
Fraud Rate Hex Color = 
SWITCH(
    [Fraud Rate RAG Status],
    "RED", "#E74C3C",
    "AMBER", "#F39C12",
    "GREEN", "#2ECC71",
    "#BDC3C7"
)
```

### `[Provider Risk Badge Color]`
Conditional color code based on individual provider model probability.
```dax
Provider Risk Badge Color = 
SWITCH(
    SELECTEDVALUE('provider_risk_scores'[risk_tier]),
    "High Risk", "#D9534F",
    "Medium Risk", "#F0AD4E",
    "Low Risk", "#5CB85C",
    "#CCCCCC"
)
```

---

## 4. Time Intelligence & Trend Measures

### `[Prior Month Claims]`
Total claims volume from the previous month.
```dax
Prior Month Claims = 
CALCULATE(
    SUM('temporal_trends'[total_claims]),
    DATEADD('temporal_trends'[claim_year_month], -1, MONTH)
)
```

### `[MoM Claim Volume Growth %]`
Month-over-Month percentage change in claims volume.
```dax
MoM Claim Volume Growth % = 
VAR CurrentMonth = SUM('temporal_trends'[total_claims])
VAR PriorMonth = [Prior Month Claims]
RETURN
    IF(
        ISBLANK(PriorMonth),
        BLANK(),
        DIVIDE(CurrentMonth - PriorMonth, PriorMonth, 0) * 100
    )
```

### `[3-Month Rolling Avg Reimbursement]`
Smoothed rolling 3-month reimbursement trend to filter seasonal anomalies.
```dax
3-Month Rolling Avg Reimbursement = 
AVERAGEX(
    DATESINPERIOD(
        'temporal_trends'[claim_year_month],
        LASTDATE('temporal_trends'[claim_year_month]),
        -3,
        MONTH
    ),
    CALCULATE(SUM('temporal_trends'[total_reimbursement]))
)
```

---

## 5. Drill-Through & Dynamic Context Measures

### `[Provider State Reimbursement Rank]`
Ranks the selected provider within their operating state by total payout.
```dax
Provider State Reimbursement Rank = 
RANKX(
    ALLSELECTED('provider_risk_scores'),
    [Total Reimbursement],
    ,
    DESC,
    Dense
)
```

### `[Selected Provider Risk Summary]`
Dynamic text string summarizing the selected provider for case reviewers.
```dax
Selected Provider Risk Summary = 
VAR ProvID = SELECTEDVALUE('provider_risk_scores'[provider_id], "None Selected")
VAR Prob = SELECTEDVALUE('provider_risk_scores'[fraud_risk_probability], 0)
VAR Tier = SELECTEDVALUE('provider_risk_scores'[risk_tier], "Unknown")
RETURN
    "Provider " & ProvID & " | Model Probability: " & FORMAT(Prob, "0.0%") & " | Tier: " & Tier
```
