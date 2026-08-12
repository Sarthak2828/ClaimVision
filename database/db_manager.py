"""
ClaimVision Canonical PostgreSQL Database Manager.

Connects to PostgreSQL using DATABASE_URL from environment or .env file.
Canonical database: PostgreSQL 16 via Docker Compose (see docker-compose.yml).
A developer with an existing PostgreSQL installation may point DATABASE_URL to it.

Why PostgreSQL?
ClaimVision uses a relational healthcare claims star-schema with foreign keys,
joins, CTEs, window functions, aggregations, and EXPLAIN ANALYZE query plans.
PostgreSQL provides a full-featured relational environment for demonstrating
these concepts reproducibly via Docker Compose.
"""

import os
import logging
from pathlib import Path
from typing import Optional
import sqlalchemy as sa
from sqlalchemy import text
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages PostgreSQL connection lifecycle for ClaimVision.

    Connection is resolved in priority order:
      1. Explicit connection_url argument
      2. DATABASE_URL environment variable
      3. DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD environment variables

    PostgreSQL 16 via Docker Compose is the canonical environment.
    Run:  docker compose up -d   before executing the pipeline.
    """

    def __init__(self, connection_url: Optional[str] = None):
        load_dotenv()

        if connection_url:
            self.url = connection_url
        else:
            db_url = os.getenv("DATABASE_URL", "")
            if db_url and "postgres" in db_url:
                if db_url.startswith("postgresql://") and "+psycopg2" not in db_url:
                    self.url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
                else:
                    self.url = db_url
            else:
                host = os.getenv("DB_HOST", "localhost")
                port = os.getenv("DB_PORT", "5432")
                name = os.getenv("DB_NAME", "claimvision_db")
                user = os.getenv("DB_USER", "postgres")
                pw = os.getenv("DB_PASSWORD", "postgrespassword")
                self.url = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{name}"

        logger.info(f"Connecting to PostgreSQL at: {self.url.split('@')[-1]}")
        self.engine = sa.create_engine(
            self.url,
            echo=False,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 10},
        )
        try:
            with self.engine.connect() as conn:
                ver = conn.execute(text("SELECT version()")).scalar()
                logger.info(f"PostgreSQL connected: {ver}")
        except Exception as e:
            logger.error(
                f"Cannot connect to PostgreSQL: {e}\n"
                "Ensure Docker Compose is running: docker compose up -d\n"
                "Or set DATABASE_URL in your .env file."
            )
            raise

    def execute_ddl(self, schema_file_path: Optional[Path] = None) -> None:
        """Execute native PostgreSQL DDL script."""
        if schema_file_path is None:
            schema_file_path = Path(__file__).resolve().parent / "schema.sql"

        with open(schema_file_path, "r", encoding="utf-8") as f:
            ddl_sql = f.read()

        statements = [s.strip() for s in ddl_sql.split(";") if s.strip()]
        with self.engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
        logger.info(f"Successfully executed PostgreSQL DDL from {schema_file_path.name}")

    def get_table_counts(self) -> dict:
        """Return row counts for all core schema tables."""
        tables = [
            "dim_providers", "dim_beneficiaries", "fact_inpatient_claims",
            "fact_outpatient_claims", "fact_claims_unified"
        ]
        counts = {}
        with self.engine.connect() as conn:
            for tbl in tables:
                try:
                    counts[tbl] = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
                except Exception:
                    counts[tbl] = None
        return counts


if __name__ == "__main__":
    db = DatabaseManager()
    db.execute_ddl()
    print("PostgreSQL Schema initialized. Table counts:", db.get_table_counts())
