from utils.db_utils import get_sqlalchemy_engine, read_table
import pandas as pd

engine = get_sqlalchemy_engine()
bl = read_table("demand_block_actions", engine=engine)
bl["week_start"] = pd.to_datetime(bl["week_start"])
recent = bl[bl["week_start"] == "2026-02-10"]
print(f"Total recent blocks: {len(recent)}")
print(f"block_reason null: {recent['block_reason'].isna().sum()}")
print(f"block_reason non-null: {recent['block_reason'].notna().sum()}")
print(recent[["block_reason"]].head(5))
