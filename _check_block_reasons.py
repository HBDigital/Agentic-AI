"""Check block_reason values in Demand_Block_Actions."""
import sys
sys.path.insert(0, ".")
from utils.db_utils import get_sqlalchemy_engine
import pandas as pd

engine = get_sqlalchemy_engine()
df = pd.read_sql("SELECT block_id, property_id, CAST(block_start_date AS DATE) as bsd, block_reason FROM Demand_Block_Actions ORDER BY bsd, property_id", engine)
print(f"Total rows: {len(df)}")
print(f"Rows with NULL block_reason: {df['block_reason'].isna().sum()}")
print(f"Rows with non-NULL block_reason: {df['block_reason'].notna().sum()}")
print()

# Show sample of NULL rows
null_rows = df[df['block_reason'].isna()]
if not null_rows.empty:
    print("Sample NULL block_reason rows:")
    print(null_rows.head(20).to_string(index=False))
    print()
    print("Weeks with NULL block_reason:")
    print(null_rows.groupby('bsd').size().to_string())

print()
# Show sample of non-NULL rows
notnull = df[df['block_reason'].notna()]
if not notnull.empty:
    print("\nSample non-NULL block_reason rows:")
    print(notnull.head(10).to_string(index=False))
