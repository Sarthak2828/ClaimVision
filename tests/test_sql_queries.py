"""
Unit tests verifying SQL query syntax and execution against the analytical database.
"""

import pytest
from sqlalchemy import text
from database.db_manager import DatabaseManager


@pytest.fixture(scope="module")
def db():
    return DatabaseManager()


def test_sql_kpi_snapshot_query(db):
    query = """
        SELECT
            COUNT(*) AS total_claims,
            ROUND(SUM(reimbursed_amount), 2) AS total_reimbursement_usd
        FROM fact_claims_unified;
    """
    with db.engine.connect() as conn:
        res = conn.execute(text(query)).mappings().one()
        assert res["total_claims"] > 0
        assert res["total_reimbursement_usd"] > 0


def test_sql_inpatient_vs_outpatient_query(db):
    query = """
        SELECT claim_type, COUNT(*) AS count
        FROM fact_claims_unified
        GROUP BY claim_type;
    """
    with db.engine.connect() as conn:
        rows = conn.execute(text(query)).mappings().all()
        types = {r["claim_type"] for r in rows}
        assert "Inpatient" in types
        assert "Outpatient" in types


def test_sql_window_function_dense_rank(db):
    query = """
        WITH ProviderTotals AS (
            SELECT
                c.provider_id,
                b.state_id,
                SUM(c.reimbursed_amount) AS total_amount
            FROM fact_claims_unified c
            JOIN dim_beneficiaries b ON c.bene_id = b.bene_id
            GROUP BY c.provider_id, b.state_id
        )
        SELECT
            provider_id,
            state_id,
            DENSE_RANK() OVER (PARTITION BY state_id ORDER BY total_amount DESC) AS prov_rank
        FROM ProviderTotals
        LIMIT 10;
    """
    with db.engine.connect() as conn:
        rows = conn.execute(text(query)).mappings().all()
        assert len(rows) > 0
        assert "prov_rank" in rows[0]
