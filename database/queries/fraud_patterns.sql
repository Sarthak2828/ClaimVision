-- =====================================================================
-- Section 4: Suspicious Fraud Patterns & Behavioral Profiling (PostgreSQL)
-- =====================================================================

-- Q14: Phantom Billing: Same Beneficiary Billed Multiple Times on Same Date
SELECT
    provider_id,
    bene_id,
    claim_start_date,
    COUNT(claim_id) AS same_day_claim_count,
    ROUND(SUM(reimbursed_amount)::numeric, 2) AS total_same_day_payout
FROM fact_claims_unified
GROUP BY provider_id, bene_id, claim_start_date
HAVING COUNT(claim_id) > 1
ORDER BY same_day_claim_count DESC, total_same_day_payout DESC
LIMIT 15;

-- Q15: Inpatient Stay Outliers (>15 Days with Disproportionate Billing)
SELECT
    c.claim_id,
    c.provider_id,
    p.fraud_label,
    c.length_of_stay,
    c.reimbursed_amount,
    ROUND((c.reimbursed_amount / NULLIF(c.length_of_stay, 0))::numeric, 2) AS reimbursement_per_day,
    c.admit_diagnosis_code
FROM fact_inpatient_claims c
JOIN dim_providers p ON c.provider_id = p.provider_id
WHERE c.length_of_stay >= 15
ORDER BY c.reimbursed_amount DESC
LIMIT 15;

-- Q16: Physician Multiplicity / Provider Churn Ratio
-- Fraudulent providers frequently bill under rotating attending physician IDs
SELECT
    c.provider_id,
    p.fraud_label,
    COUNT(DISTINCT c.claim_id) AS total_claims,
    COUNT(DISTINCT c.attending_physician) AS distinct_physicians,
    ROUND(CAST(COUNT(DISTINCT c.attending_physician) AS NUMERIC) / COUNT(DISTINCT c.claim_id), 3) AS physician_to_claim_ratio
FROM fact_claims_unified c
JOIN dim_providers p ON c.provider_id = p.provider_id
GROUP BY c.provider_id, p.fraud_label
HAVING COUNT(DISTINCT c.claim_id) >= 50
ORDER BY physician_to_claim_ratio DESC
LIMIT 15;

-- Q17: High-Cost Diagnosis Code Concentration
SELECT
    primary_diagnosis_code,
    COUNT(*) AS diagnosis_occurrences,
    ROUND(SUM(reimbursed_amount)::numeric, 2) AS total_amount,
    ROUND(AVG(reimbursed_amount)::numeric, 2) AS avg_claim_cost
FROM fact_claims_unified
WHERE primary_diagnosis_code IS NOT NULL
GROUP BY primary_diagnosis_code
ORDER BY total_amount DESC
LIMIT 10;
