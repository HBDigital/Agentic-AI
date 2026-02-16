"""Debug: simulate exactly what the dashboard does for week 2026-03-02."""
import sys
sys.path.insert(0, ".")
from utils.db_utils import get_sqlalchemy_engine, read_table
import pandas as pd

# Load exactly as dashboard does
bl_all = read_table("demand_block_actions")
print(f"Total rows: {len(bl_all)}")
print(f"Columns: {list(bl_all.columns)}")
print(f"block_reason dtype: {bl_all['block_reason'].dtype}")
print(f"block_reason NULLs: {bl_all['block_reason'].isna().sum()}")
print()

# Filter exactly as dashboard does
bl_all["week_start"] = pd.to_datetime(bl_all["week_start"])
bl_all["block_start_date"] = pd.to_datetime(bl_all["block_start_date"])

ws = "2026-03-02"
bl = bl_all[bl_all["week_start"] == pd.to_datetime(ws)]
print(f"Rows for week {ws}: {len(bl)}")

if bl.empty:
    print("EMPTY! Falling back to all weeks.")
    bl = bl_all

print(f"block_reason NULLs in filtered: {bl['block_reason'].isna().sum()}")
print(f"block_reason sample:")
print(bl[['property_id', 'week_start', 'block_reason']].head(25).to_string(index=False))

# Simulate the reason splitting
reasons_col = bl["block_reason"].fillna("Composite score above threshold")
rr = []
for r in reasons_col:
    rr.extend([x.strip() for x in str(r).split("+")])
rc = pd.Series(rr).value_counts().head(10).reset_index()
rc.columns = ["Reason", "Count"]
print()
print("Reason chart data:")
print(rc.to_string(index=False))
