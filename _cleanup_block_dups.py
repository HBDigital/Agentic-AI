"""One-time script to remove duplicate blocking rows from Demand_Block_Actions."""
import sys
sys.path.insert(0, ".")
from utils.db_utils import get_sqlalchemy_engine
from sqlalchemy import text

engine = get_sqlalchemy_engine()
with engine.connect() as conn:
    # Count before
    before = conn.execute(text("SELECT COUNT(*) FROM Demand_Block_Actions")).fetchone()[0]
    print(f"Before: {before} rows")

    # Delete duplicates, keeping MIN(block_id) per property+week
    result = conn.execute(text("""
        DELETE FROM Demand_Block_Actions
        WHERE block_id NOT IN (
            SELECT MIN(block_id)
            FROM Demand_Block_Actions
            GROUP BY property_id, CAST(block_start_date AS DATE)
        )
    """))
    conn.commit()
    print(f"Deleted: {result.rowcount} duplicate rows")

    # Count after
    after = conn.execute(text("SELECT COUNT(*) FROM Demand_Block_Actions")).fetchone()[0]
    print(f"After: {after} rows")
