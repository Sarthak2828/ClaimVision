-- =====================================================================
-- Section 2: Joins, CTEs & Subqueries
-- =====================================================================

-- Q6: Comprehensive 3-Way Relational Join with Demographics & Risk
SELECT
    c.claim_id,
    c.claim_type,
    c.provider_id,
    p.fraud_label,
    b.bene_id,
    b.gender,
    b.state_id,
    c.claim_start_date,
    c.reimbursed_amount,
    c.length_of_stay
FROM fact_claims_unified c
JOIN dim_providers p ON c.provider_id = p.provider_id
JOIN dim_beneficiaries b ON c.bene_id = b.bene_id
WHERE c.reimbursed_amount > 5000
ORDER BY c.reimbursed_amount DESC
LIMIT 20;

-- Q7: Multi-CTE Pipeline: High-Cost Chronic Illness Comorbidities vs Fraud Exposure
WITH PatientChronicProfile AS (
    SELECT
        bene_id,
        (chronic_alzheimer + chronic_heartfailure + chronic_kidneydisease + 
         chronic_cancer + chronic_copd + chronic_depression + chronic_diabetes + 
         chronic_stroke) AS total_chronic_conditions
    FROM dim_beneficiaries
),
ClaimEnriched AS (
    SELECT
        c.claim_id,
        c.provider_id,
        p.fraud_label,
        c.reimbursed_amount,
        pcp.total_chronic_conditions
    FROM fact_claims_unified c
    JOIN dim_providers p ON c.provider_id = p.provider_id
    JOIN PatientChronicProfile pcp ON c.bene_id = pcp.bene_id
)
SELECT
    fraud_label,
    ROUND(AVG(total_chronic_conditions), 2) AS avg_patient_chronic_conditions,
    ROUND(AVG(reimbursed_amount), 2) AS avg_reimbursement,
    COUNT(*) AS total_claims
FROM ClaimEnriched
GROUP BY fraud_label;

-- Q8: Data Integrity Verification via LEFT JOIN (Identify Orphan Claims)
SELECT
    c.claim_id,
    c.provider_id,
    c.bene_id
FROM fact_claims_unified c
LEFT JOIN dim_providers p ON c.provider_id = p.provider_id
LEFT JOIN dim_beneficiaries b ON c.bene_id = b.bene_id
WHERE p.provider_id IS NULL OR b.bene_id IS NULL;

-- Q9: Subquery: Providers Exceeding 2x Their State Average Inpatient Length of Stay
SELECT
    prov.provider_id,
    prov.state_id,
    prov.avg_prov_los,
    state_avg.avg_state_los
FROM (
    SELECT
        c.provider_id,
        b.state_id,
        ROUND(AVG(c.length_of_stay), 2) AS avg_prov_los
    FROM fact_inpatient_claims c
    JOIN dim_beneficiaries b ON c.bene_id = b.bene_id
    GROUP BY c.provider_id, b.state_id
    HAVING COUNT(*) >= 10
) prov
JOIN (
    SELECT
        b.state_id,
        ROUND(AVG(c.length_of_stay), 2) AS avg_state_los
    FROM fact_inpatient_claims c
    JOIN dim_beneficiaries b ON c.bene_id = b.bene_id
    GROUP BY b.state_id
) state_avg ON prov.state_id = state_avg.state_id
WHERE prov.avg_prov_los > (2.0 * state_avg.avg_state_los)
ORDER BY prov.avg_prov_los DESC
LIMIT 15;
