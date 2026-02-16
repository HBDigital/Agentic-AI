"""Check which block_reason rows are NULL and for which weeks."""
import sys
sys.path.insert(0, ".")
from utils.db_utils import get_sqlalchemy_engine
import pandas as pd

engine = get_sqlalchemy_engine()
df = pd.read_sql("SELECT block_id, property_id, CAST(week_start AS DATE) as ws, CAST(block_start_date AS DATE) as bsd, block_reason FROM Demand_Block_Actions ORDER BY ws, property_id", engine)

print(f"Total rows: {len(df)}")
null_rows = df[df['block_reason'].isna()]
notnull_rows = df[df['block_reason'].notna()]
print(f"NULL block_reason: {len(null_rows)}")
print(f"Non-NULL block_reason: {len(notnull_rows)}")

if not null_rows.empty:
    print("\nWeeks with NULL block_reason:")
    print(null_rows.groupby('ws').size().to_string())
    print("\nSample NULL rows:")
    print(null_rows.head(20).to_string(index=False))

# Check the selected week
for w in ['2026-02-10', '2026-02-16', '2026-02-23', '2026-03-02']:
    wk = df[df['ws'] == pd.to_datetime(w).date()]
    nulls = wk['block_reason'].isna().sum()
    print(f"\nWeek {w}: {len(wk)} rows, {nulls} NULL block_reason")
