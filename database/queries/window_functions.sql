-- =====================================================================
-- Section 3: Window Functions & Analytical Partitioning
-- =====================================================================

-- Q10: Provider Reimbursement Ranking Within Each State (DENSE_RANK)
WITH ProviderStateTotals AS (
    SELECT
        b.state_id,
        c.provider_id,
        p.fraud_label,
        ROUND(SUM(c.reimbursed_amount), 2) AS total_state_reimbursement
    FROM fact_claims_unified c
    JOIN dim_beneficiaries b ON c.bene_id = b.bene_id
    JOIN dim_providers p ON c.provider_id = p.provider_id
    GROUP BY b.state_id, c.provider_id, p.fraud_label
)
SELECT
    state_id,
    provider_id,
    fraud_label,
    total_state_reimbursement,
    DENSE_RANK() OVER (PARTITION BY state_id ORDER BY total_state_reimbursement DESC) AS state_payout_rank
FROM ProviderStateTotals
WHERE state_id IN (1, 5, 10, 33)
ORDER BY state_id ASC, state_payout_rank ASC
LIMIT 20;

-- Q11: Cumulative Running Total of Claims Payout Across Months
WITH MonthlySummary AS (
    SELECT
        SUBSTR(claim_start_date, 1, 7) AS claim_month,
        ROUND(SUM(reimbursed_amount), 2) AS monthly_payout
    FROM fact_claims_unified
    GROUP BY SUBSTR(claim_start_date, 1, 7)
)
SELECT
    claim_month,
    monthly_payout,
    SUM(monthly_payout) OVER (ORDER BY claim_month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_payout_ytd
FROM MonthlySummary
ORDER BY claim_month ASC;

-- Q12: Provider Risk Deciles via NTILE(10)
WITH ProviderVolume AS (
    SELECT
        provider_id,
        COUNT(*) AS total_claims,
        ROUND(SUM(reimbursed_amount), 2) AS total_reimbursement
    FROM fact_claims_unified
    GROUP BY provider_id
)
SELECT
    provider_id,
    total_claims,
    total_reimbursement,
    NTILE(10) OVER (ORDER BY total_reimbursement DESC) AS payout_decile
FROM ProviderVolume
LIMIT 20;

-- Q13: Moving Average (3-claim rolling average reimbursement per provider)
SELECT
    claim_id,
    provider_id,
    claim_start_date,
    reimbursed_amount,
    ROUND(AVG(reimbursed_amount) OVER (
        PARTITION BY provider_id 
        ORDER BY claim_start_date, claim_id
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2) AS rolling_3claim_avg_amount
FROM fact_claims_unified
WHERE provider_id IN (SELECT provider_id FROM dim_providers WHERE potential_fraud = 1 LIMIT 3)
ORDER BY provider_id, claim_start_date
LIMIT 25;
