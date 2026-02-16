"""Analyze blocking score distribution to understand why 172/200 properties are blocked."""
import sys
sys.path.insert(0, ".")
from utils.db_utils import get_sqlalchemy_engine
import pandas as pd

engine = get_sqlalchemy_engine()

# Check current blocks for week 2026-02-10
df = pd.read_sql("""
    SELECT property_id, rooms_blocked_per_night, expected_revenue_uplift_inr, block_reason
    FROM Demand_Block_Actions 
    WHERE CAST(block_start_date AS DATE) = '2026-02-10'
    ORDER BY expected_revenue_uplift_inr DESC
""", engine)

print(f"Total properties blocked for 2026-02-10: {len(df)}")
print(f"Total properties: 200")
print(f"Block rate: {len(df)/200*100:.0f}%")
print()

# Room distribution
print("Rooms blocked distribution:")
print(df['rooms_blocked_per_night'].describe())
print()

# Reason breakdown
print("Block reasons breakdown:")
for reason in ['High sellout risk', 'Demand spike', 'Price inflation', 'Event activity', 'Composite score']:
    count = df['block_reason'].str.contains(reason, na=False).sum()
    print(f"  {reason}: {count} properties")

print()
print("Top 10 by uplift:")
print(df.head(10).to_string(index=False))
print()
print("Bottom 10 by uplift:")
print(df.tail(10).to_string(index=False))
