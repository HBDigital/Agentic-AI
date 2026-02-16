"""Check and clean blocks for the 3 target weeks so fresh event-aware blocks can be generated."""
import sys
sys.path.insert(0, ".")
from utils.db_utils import get_sqlalchemy_engine
from sqlalchemy import text

engine = get_sqlalchemy_engine()
with engine.connect() as conn:
    # Count total
    total = conn.execute(text("SELECT COUNT(*) FROM Demand_Block_Actions")).fetchone()[0]
    print(f"Total rows: {total}")

    # Count for target weeks
    weeks = conn.execute(text(
        "SELECT CAST(block_start_date AS DATE) as bsd, COUNT(*) as cnt "
        "FROM Demand_Block_Actions "
        "WHERE CAST(block_start_date AS DATE) IN ('2026-02-10','2026-02-16','2026-02-23','2026-03-02') "
        "GROUP BY CAST(block_start_date AS DATE) ORDER BY bsd"
    )).fetchall()
    for w in weeks:
        print(f"  Week {w[0]}: {w[1]} rows")

    target_count = conn.execute(text(
        "SELECT COUNT(*) FROM Demand_Block_Actions "
        "WHERE CAST(block_start_date AS DATE) IN ('2026-02-10','2026-02-16','2026-02-23','2026-03-02')"
    )).fetchone()[0]
    print(f"\nTotal rows to DELETE for these 4 weeks: {target_count}")
    print(f"Rows remaining after delete: {total - target_count}")

    if "--delete" in sys.argv:
        result = conn.execute(text(
            "DELETE FROM Demand_Block_Actions "
            "WHERE CAST(block_start_date AS DATE) IN ('2026-02-10','2026-02-16','2026-02-23','2026-03-02')"
        ))
        conn.commit()
        print(f"\nDeleted {result.rowcount} rows")
        after = conn.execute(text("SELECT COUNT(*) FROM Demand_Block_Actions")).fetchone()[0]
        print(f"Remaining: {after} rows")
    else:
        print("\nRun with --delete to actually remove them.")
