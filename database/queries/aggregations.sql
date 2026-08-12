-- =====================================================================
-- Section 1: Aggregations & Business KPIs
-- =====================================================================

-- Q1: Portfolio-Wide Executive KPI Snapshot
SELECT
    COUNT(*) AS total_claims,
    COUNT(DISTINCT bene_id) AS distinct_beneficiaries,
    COUNT(DISTINCT provider_id) AS active_providers,
    ROUND(SUM(reimbursed_amount), 2) AS total_reimbursement_usd,
    ROUND(AVG(reimbursed_amount), 2) AS avg_reimbursement_usd,
    ROUND(SUM(deductible_amount), 2) AS total_deductible_usd
FROM fact_claims_unified;

-- Q2: Financial Breakdown by Claim Type (Inpatient vs Outpatient)
SELECT
    claim_type,
    COUNT(*) AS claim_count,
    ROUND(SUM(reimbursed_amount), 2) AS total_reimbursed,
    ROUND(AVG(reimbursed_amount), 2) AS avg_reimbursed,
    ROUND(AVG(deductible_amount), 2) AS avg_deductible,
    ROUND(AVG(length_of_stay), 1) AS avg_length_of_stay_days
FROM fact_claims_unified
GROUP BY claim_type;

-- Q3: High-Volume, High-Reimbursement Providers (HAVING clause)
SELECT
    c.provider_id,
    p.fraud_label,
    COUNT(*) AS claim_count,
    ROUND(SUM(c.reimbursed_amount), 2) AS total_payout,
    ROUND(AVG(c.reimbursed_amount), 2) AS avg_payout_per_claim
FROM fact_claims_unified c
JOIN dim_providers p ON c.provider_id = p.provider_id
GROUP BY c.provider_id, p.fraud_label
HAVING COUNT(*) >= 100 AND AVG(c.reimbursed_amount) > 1500
ORDER BY total_payout DESC
LIMIT 15;

-- Q4: Monthly Claim Volume and Financial Exposure Trend
SELECT
    SUBSTR(claim_start_date, 1, 7) AS claim_year_month,
    COUNT(*) AS monthly_claims,
    ROUND(SUM(reimbursed_amount), 2) AS monthly_reimbursed_usd,
    ROUND(AVG(reimbursed_amount), 2) AS avg_claim_cost
FROM fact_claims_unified
GROUP BY SUBSTR(claim_start_date, 1, 7)
ORDER BY claim_year_month ASC;

-- Q5: Geographic Claim Volume and Fraud Breakdown by State
SELECT
    b.state_id,
    COUNT(c.claim_id) AS total_claims,
    COUNT(DISTINCT c.provider_id) AS providers_operating,
    ROUND(SUM(c.reimbursed_amount), 2) AS total_reimbursement,
    ROUND(100.0 * SUM(CASE WHEN p.potential_fraud = 1 THEN 1 ELSE 0 END) / COUNT(c.claim_id), 2) AS fraud_claim_percentage
FROM fact_claims_unified c
JOIN dim_beneficiaries b ON c.bene_id = b.bene_id
JOIN dim_providers p ON c.provider_id = p.provider_id
GROUP BY b.state_id
ORDER BY total_claims DESC
LIMIT 10;
