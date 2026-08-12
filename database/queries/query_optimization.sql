-- =====================================================================
-- Section 5: Query Optimization & EXPLAIN Benchmarking
-- =====================================================================

-- Target Query: Provider aggregation filtering on high reimbursement
-- Demonstrates query performance benefits of idx_unified_provider_amount

-- EXPLAIN QUERY PLAN (Index Scan vs Full Table Scan):
EXPLAIN QUERY PLAN
SELECT
    provider_id,
    COUNT(*) AS claim_count,
    ROUND(SUM(reimbursed_amount), 2) AS total_reimbursed
FROM fact_claims_unified
WHERE reimbursed_amount > 2000
GROUP BY provider_id
ORDER BY total_reimbursed DESC
LIMIT 10;
