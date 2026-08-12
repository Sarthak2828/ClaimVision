"""
ClaimVision SQL Query Execution Runner & Benchmarking Utility.
Executes all SQL query modules, displays result sets, and measures execution timings.
"""

import time
import logging
from pathlib import Path
import pandas as pd
from sqlalchemy import text
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from database.db_manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_sql_file(db: DatabaseManager, file_path: Path) -> None:
    """Read a SQL file, split into individual statements, execute and profile."""
    logger.info(f"\n{'='*70}\nExecuting SQL Module: {file_path.name}\n{'='*70}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_statements = content.split(";")
    query_num = 1
    with db.engine.connect() as conn:
        for raw_stmt in raw_statements:
            clean_stmt = "\n".join(
                line for line in raw_stmt.splitlines() if not line.strip().startswith("--")
            ).strip()
            if not clean_stmt:
                continue

            logger.info(f"--- Running Query #{query_num} ---")
            start_time = time.perf_counter()
            try:
                res = conn.execute(text(clean_stmt))
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if res.returns_rows:
                    df = pd.DataFrame(res.fetchall(), columns=res.keys())
                    print(f"Result (Execution Time: {elapsed_ms:.2f} ms | Rows: {len(df)}):")
                    print(df.head(5).to_string(index=False))
                else:
                    print(f"Executed statement in {elapsed_ms:.2f} ms")
            except Exception as e:
                logger.error(f"Error executing Query #{query_num}: {e}")
            query_num += 1


def run_all_queries() -> None:
    """Execute all analytical SQL suites."""
    db = DatabaseManager()
    query_dir = Path(__file__).resolve().parent / "queries"
    sql_files = [
        "aggregations.sql",
        "joins_subqueries.sql",
        "window_functions.sql",
        "fraud_patterns.sql",
        "query_optimization.sql"
    ]
    for filename in sql_files:
        run_sql_file(db, query_dir / filename)


if __name__ == "__main__":
    run_all_queries()
