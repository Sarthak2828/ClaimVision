"""
ClaimVision Test Suite — Session-Scoped Database Fixture
=========================================================

This conftest.py ensures that when PostgreSQL is reachable, the canonical
star-schema and a minimal set of deterministic synthetic rows exist BEFORE
any database-dependent test runs.

WHY THIS IS NEEDED
------------------
The DDL and data are normally created by the ETL pipeline (run_pipeline.py).
In CI, the PostgreSQL service container starts empty. If pytest runs before
the pipeline (or in isolation), tables do not exist and SQL tests fail with
"relation does not exist".

This fixture makes the test suite self-contained: it uses the real schema.sql
DDL (same as production) and inserts just enough rows for every SQL test to
pass its assertions.

If PostgreSQL is not reachable, the fixture does nothing — individual tests
will skip via their own guards.
If tables already exist and have data (e.g., after a local pipeline run), the
fixture skips initialisation so real data is tested instead.
"""

import os
import sys
from pathlib import Path
import pytest

# Ensure src/ and database/ are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Minimal synthetic test dataset
# ---------------------------------------------------------------------------
# - 4 providers (2 fraud, 2 clean) across 2 states
# - 3 beneficiaries, spread across state 1 and state 2
# - 3 inpatient claims + 4 outpatient claims
# - 7 unified claims (both types present)
# These numbers are sufficient for every assertion in test_sql_queries.py

_PROVIDERS = [
    ("PRV00001", True,  "Yes"),
    ("PRV00002", False, "No"),
    ("PRV00003", True,  "Yes"),
    ("PRV00004", False, "No"),
]

_BENEFICIARIES = [
    # (bene_id, dob, gender, race, state_id, county_id)
    ("BENE0001", "1950-01-01", 1, 1, 1, 10),
    ("BENE0002", "1960-05-15", 2, 2, 1, 10),
    ("BENE0003", "1945-11-20", 1, 1, 2, 20),
]

_INPATIENT = [
    # (claim_id, bene_id, provider_id, start, end, adm, dis, los, reimb, deduct, phys)
    ("ICLM0001", "BENE0001", "PRV00001", "2009-01-10", "2009-01-15",
     "2009-01-10", "2009-01-15", 5, 12000.00, 1000.00, "PHY001"),
    ("ICLM0002", "BENE0002", "PRV00001", "2009-02-01", "2009-02-08",
     "2009-02-01", "2009-02-08", 7, 18000.00, 1500.00, "PHY002"),
    ("ICLM0003", "BENE0003", "PRV00003", "2009-03-05", "2009-03-10",
     "2009-03-05", "2009-03-10", 5, 9500.00,  800.00,  "PHY003"),
]

_OUTPATIENT = [
    # (claim_id, bene_id, provider_id, start, end, reimb, deduct, phys)
    ("OCLM0001", "BENE0001", "PRV00002", "2009-01-20", "2009-01-20", 350.00,  50.00,  "PHY004"),
    ("OCLM0002", "BENE0002", "PRV00002", "2009-02-10", "2009-02-10", 420.00,  75.00,  "PHY004"),
    ("OCLM0003", "BENE0003", "PRV00004", "2009-03-15", "2009-03-15", 680.00,  100.00, "PHY005"),
    ("OCLM0004", "BENE0001", "PRV00004", "2009-04-01", "2009-04-01", 290.00,  40.00,  "PHY005"),
]

# Unified = inpatient rows + outpatient rows
_UNIFIED_INPATIENT = [
    # (claim_id, claim_type, bene_id, provider_id, start, end, reimb, deduct, los, phys)
    ("ICLM0001", "Inpatient", "BENE0001", "PRV00001", "2009-01-10", "2009-01-15", 12000.00, 1000.00, 5, "PHY001"),
    ("ICLM0002", "Inpatient", "BENE0002", "PRV00001", "2009-02-01", "2009-02-08", 18000.00, 1500.00, 7, "PHY002"),
    ("ICLM0003", "Inpatient", "BENE0003", "PRV00003", "2009-03-05", "2009-03-10",  9500.00,  800.00, 5, "PHY003"),
]
_UNIFIED_OUTPATIENT = [
    ("OCLM0001", "Outpatient", "BENE0001", "PRV00002", "2009-01-20", "2009-01-20",  350.00,  50.00, 0, "PHY004"),
    ("OCLM0002", "Outpatient", "BENE0002", "PRV00002", "2009-02-10", "2009-02-10",  420.00,  75.00, 0, "PHY004"),
    ("OCLM0003", "Outpatient", "BENE0003", "PRV00004", "2009-03-15", "2009-03-15",  680.00, 100.00, 0, "PHY005"),
    ("OCLM0004", "Outpatient", "BENE0001", "PRV00004", "2009-04-01", "2009-04-01",  290.00,  40.00, 0, "PHY005"),
]


def _tables_are_populated(conn) -> bool:
    """Return True if the schema has been initialized and has data."""
    from sqlalchemy import text
    try:
        count = conn.execute(
            text("SELECT COUNT(*) FROM fact_claims_unified")
        ).scalar()
        return count > 0
    except Exception:
        return False


def _initialize_schema(engine) -> None:
    """Run schema.sql DDL (DROP + CREATE tables + indexes)."""
    from sqlalchemy import text
    schema_path = Path(__file__).resolve().parent.parent / "database" / "schema.sql"
    ddl = schema_path.read_text(encoding="utf-8")
    # Execute each statement individually (split on ";" to handle multi-statement DDL)
    statements = [s.strip() for s in ddl.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def _load_test_data(engine) -> None:
    """Insert minimal deterministic synthetic rows into all five tables."""
    from sqlalchemy import text

    with engine.begin() as conn:
        # 1. dim_providers
        for prov_id, fraud_bool, fraud_label in _PROVIDERS:
            conn.execute(text(
                "INSERT INTO dim_providers (provider_id, potential_fraud, fraud_label) "
                "VALUES (:pid, :fb, :fl) ON CONFLICT DO NOTHING"
            ), {"pid": prov_id, "fb": fraud_bool, "fl": fraud_label})

        # 2. dim_beneficiaries
        for bene_id, dob, gender, race, state_id, county_id in _BENEFICIARIES:
            conn.execute(text(
                "INSERT INTO dim_beneficiaries "
                "(bene_id, dob, gender, race, state_id, county_id) "
                "VALUES (:bid, :dob, :g, :r, :s, :c) ON CONFLICT DO NOTHING"
            ), {"bid": bene_id, "dob": dob, "g": gender, "r": race,
                "s": state_id, "c": county_id})

        # 3. fact_inpatient_claims
        for (cid, bid, pid, start, end, adm, dis,
             los, reimb, deduct, phys) in _INPATIENT:
            conn.execute(text(
                "INSERT INTO fact_inpatient_claims "
                "(claim_id, bene_id, provider_id, claim_start_date, claim_end_date, "
                " admission_date, discharge_date, length_of_stay, reimbursed_amount, "
                " deductible_amount, attending_physician) "
                "VALUES (:cid,:bid,:pid,:s,:e,:adm,:dis,:los,:r,:d,:phys) "
                "ON CONFLICT DO NOTHING"
            ), {"cid": cid, "bid": bid, "pid": pid, "s": start, "e": end,
                "adm": adm, "dis": dis, "los": los, "r": reimb,
                "d": deduct, "phys": phys})

        # 4. fact_outpatient_claims
        for cid, bid, pid, start, end, reimb, deduct, phys in _OUTPATIENT:
            conn.execute(text(
                "INSERT INTO fact_outpatient_claims "
                "(claim_id, bene_id, provider_id, claim_start_date, claim_end_date, "
                " reimbursed_amount, deductible_amount, attending_physician) "
                "VALUES (:cid,:bid,:pid,:s,:e,:r,:d,:phys) ON CONFLICT DO NOTHING"
            ), {"cid": cid, "bid": bid, "pid": pid, "s": start, "e": end,
                "r": reimb, "d": deduct, "phys": phys})

        # 5. fact_claims_unified (inpatient rows)
        for (cid, ctype, bid, pid, start, end,
             reimb, deduct, los, phys) in _UNIFIED_INPATIENT:
            conn.execute(text(
                "INSERT INTO fact_claims_unified "
                "(claim_id, claim_type, bene_id, provider_id, claim_start_date, "
                " claim_end_date, reimbursed_amount, deductible_amount, "
                " length_of_stay, attending_physician) "
                "VALUES (:cid,:ct,:bid,:pid,:s,:e,:r,:d,:los,:phys) "
                "ON CONFLICT DO NOTHING"
            ), {"cid": cid, "ct": ctype, "bid": bid, "pid": pid, "s": start,
                "e": end, "r": reimb, "d": deduct, "los": los, "phys": phys})

        # fact_claims_unified (outpatient rows)
        for (cid, ctype, bid, pid, start, end,
             reimb, deduct, los, phys) in _UNIFIED_OUTPATIENT:
            conn.execute(text(
                "INSERT INTO fact_claims_unified "
                "(claim_id, claim_type, bene_id, provider_id, claim_start_date, "
                " claim_end_date, reimbursed_amount, deductible_amount, "
                " length_of_stay, attending_physician) "
                "VALUES (:cid,:ct,:bid,:pid,:s,:e,:r,:d,:los,:phys) "
                "ON CONFLICT DO NOTHING"
            ), {"cid": cid, "ct": ctype, "bid": bid, "pid": pid, "s": start,
                "e": end, "r": reimb, "d": deduct, "los": los, "phys": phys})


@pytest.fixture(scope="session", autouse=True)
def ensure_test_database():
    """
    Session-scoped autouse fixture.

    When PostgreSQL is reachable and the schema/data are absent (e.g., fresh CI
    container), this fixture runs the canonical schema.sql DDL and inserts
    minimal deterministic synthetic rows so that all database-dependent tests
    can execute against a consistent baseline.

    When PostgreSQL is not reachable, the fixture does nothing — individual
    tests skip via their own guards.

    When the database is already populated (e.g., after a local pipeline run),
    the fixture does nothing so that tests run against real pipeline data.
    """
    try:
        from database.db_manager import DatabaseManager
        db = DatabaseManager()
    except Exception:
        # PostgreSQL unavailable — individual tests will skip on their own
        yield
        return

    try:
        with db.engine.connect() as conn:
            already_populated = _tables_are_populated(conn)
    except Exception:
        already_populated = False

    if not already_populated:
        _initialize_schema(db.engine)
        _load_test_data(db.engine)

    yield
    # Teardown: leave data in place for pipeline compatibility
