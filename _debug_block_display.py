"""Debug what the dashboard sees for week 2026-03-02."""
import sys
sys.path.insert(0, ".")
from utils.db_utils import get_sqlalchemy_engine
import pandas as pd

engine = get_sqlalchemy_engine()
df = pd.read_sql("SELECT * FROM Demand_Block_Actions", engine)
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()

# Check week_start values
df["week_start"] = pd.to_datetime(df["week_start"])
print("Unique week_start values:")
print(df.groupby("week_start").size().to_string())
print()

# Filter for 2026-03-02
ws = pd.to_datetime("2026-03-02")
bl = df[df["week_start"] == ws]
print(f"Rows for week 2026-03-02: {len(bl)}")
if not bl.empty:
    print(f"block_reason NULLs: {bl['block_reason'].isna().sum()}")
    print(f"block_reason sample:")
    print(bl[['property_id','block_reason']].head(10).to_string(index=False))
else:
    # Maybe week_start doesn't match - check block_start_date
    df["block_start_date"] = pd.to_datetime(df["block_start_date"])
    bl2 = df[df["block_start_date"].dt.date == ws.date()]
    print(f"Rows with block_start_date=2026-03-02: {len(bl2)}")
    if not bl2.empty:
        print(f"Their week_start values: {bl2['week_start'].unique()}")
        print(bl2[['property_id','week_start','block_start_date','block_reason']].head(10).to_string(index=False))
