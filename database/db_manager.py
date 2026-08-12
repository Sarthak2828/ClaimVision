"""
ClaimVision Database Connector & Manager.
Supports SQLite (zero setup default), DuckDB (analytical querying), and PostgreSQL.
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
    """Manages database connection lifecycle, DDL execution, and query running."""

    def __init__(self, connection_url: Optional[str] = None):
        load_dotenv()
        if connection_url:
            self.url = connection_url
        else:
            db_dialect = os.getenv("DB_DIALECT", "sqlite").lower()
            if db_dialect == "postgresql":
                host = os.getenv("DB_HOST", "localhost")
                port = os.getenv("DB_PORT", "5432")
                name = os.getenv("DB_NAME", "claimvision_db")
                user = os.getenv("DB_USER", "postgres")
                pw = os.getenv("DB_PASSWORD", "postgres")
                self.url = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{name}"
            else:
                db_path = os.getenv("DB_PATH", "data/processed/claimvision.db")
                base_dir = Path(__file__).resolve().parent.parent
                full_path = (base_dir / db_path).resolve()
                full_path.parent.mkdir(parents=True, exist_ok=True)
                self.url = f"sqlite:///{full_path}"

        self.engine = sa.create_engine(self.url, echo=False)
        logger.info(f"Initialized DatabaseManager with engine: {self.engine.dialect.name}")

    def execute_ddl(self, schema_file_path: Optional[Path] = None) -> None:
        """Execute DDL script to create clean schema."""
        if schema_file_path is None:
            schema_file_path = Path(__file__).resolve().parent / "schema.sql"

        with open(schema_file_path, "r", encoding="utf-8") as f:
            ddl_sql = f.read()

        # Split statements by semicolon for SQLite compatibility
        statements = [s.strip() for s in ddl_sql.split(";") if s.strip()]
        with self.engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
        logger.info(f"Successfully executed DDL schema from {schema_file_path.name}")

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
                    res = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
                    counts[tbl] = res
                except Exception as e:
                    counts[tbl] = None
        return counts


if __name__ == "__main__":
    db = DatabaseManager()
    db.execute_ddl()
    print("Database DDL executed. Table counts:", db.get_table_counts())
