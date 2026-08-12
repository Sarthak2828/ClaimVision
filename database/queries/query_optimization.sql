-- =====================================================================
-- Section 5: Query Optimization & EXPLAIN Benchmarking (PostgreSQL)
-- =====================================================================

-- Target Query: Provider aggregation filtering on high reimbursement
-- Demonstrates execution plan, buffer hits, and index usage in PostgreSQL:
EXPLAIN (ANALYZE, BUFFERS, COSTS)
SELECT
    provider_id,
    COUNT(*) AS claim_count,
    ROUND(SUM(reimbursed_amount)::numeric, 2) AS total_reimbursed
FROM fact_claims_unified
WHERE reimbursed_amount > 2000
GROUP BY provider_id
ORDER BY total_reimbursed DESC
LIMIT 10;
