"""
RESERVE AGENT - Execute Blocking Actions
==========================================
Responsibilities:
  1. Check real-time availability from Rate_Snapshots
  2. Select optimal supplier for block period
  3. Execute block reservations
  4. Insert records into Demand_Block_Actions table
"""
import logging
import uuid
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from utils.db_utils import get_sqlalchemy_engine, read_table, insert_dataframe, execute_query

logger = logging.getLogger(__name__)


class ReserveAgent:
    """
    Agent responsible for executing inventory blocking decisions
    by checking availability and recording block actions.
    """

    def __init__(self, engine=None, decide_agent=None):
        """
        Parameters
        ----------
        engine : sqlalchemy.Engine, optional
        decide_agent : DecideAgent
            Provides blocking decisions.
        """
        self.engine = engine or get_sqlalchemy_engine()
        self.decisions = decide_agent.get_blocking_decisions() if decide_agent else []
        self.executed_blocks: List[dict] = []
        self.failed_blocks: List[dict] = []
        # Cache table reads for performance
        self._rate_snapshots = None
        self._supplier_rel = None

    def _get_rate_snapshots(self) -> pd.DataFrame:
        if self._rate_snapshots is None:
            self._rate_snapshots = read_table("rate_snapshots", engine=self.engine)
            self._rate_snapshots["snapshot_date"] = pd.to_datetime(self._rate_snapshots["snapshot_date"])
        return self._rate_snapshots

    def _get_supplier_rel(self) -> pd.DataFrame:
        if self._supplier_rel is None:
            self._supplier_rel = read_table("supplier_reliability", engine=self.engine)
        return self._supplier_rel

    # ------------------------------------------------------------------
    # Availability Check
    # ------------------------------------------------------------------

    def check_availability(self, property_id: str, supplier_id: str,
                           start_date: str, end_date: str) -> pd.DataFrame:
        """
        Query Rate_Snapshots to verify availability for a property/supplier
        within the date range.

        Returns DataFrame of available snapshots.
        """
        rate_snapshots = read_table("rate_snapshots", engine=self.engine)
        rate_snapshots["snapshot_date"] = pd.to_datetime(rate_snapshots["snapshot_date"])

        mask = (
            (rate_snapshots["property_id"] == property_id) &
            (rate_snapshots["supplier_id"] == supplier_id) &
            (rate_snapshots["snapshot_date"] >= pd.to_datetime(start_date)) &
            (rate_snapshots["snapshot_date"] <= pd.to_datetime(end_date)) &
            (rate_snapshots["availability_flag"] == 1)
        )
        available = rate_snapshots[mask].copy()
        return available

    def find_best_supplier_option(self, property_id: str,
                                  start_date: str, end_date: str,
                                  preferred_supplier: str = None) -> Optional[dict]:
        """
        Find the best available supplier option for a block.
        Prioritizes: preferred supplier > refundable terms > lowest rate.

        Returns dict with supplier details or None if unavailable.
        """
        rate_snapshots = self._get_rate_snapshots()
        supplier_rel = self._get_supplier_rel()

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        mask = (
            (rate_snapshots["property_id"] == property_id) &
            (rate_snapshots["snapshot_date"] >= start_dt) &
            (rate_snapshots["snapshot_date"] <= end_dt) &
            (rate_snapshots["availability_flag"] == 1)
        )
        available = rate_snapshots[mask].copy()

        # If no exact matches, widen to 30 days around the block period
        if available.empty:
            wide_mask = (
                (rate_snapshots["property_id"] == property_id) &
                (rate_snapshots["snapshot_date"] >= start_dt - pd.Timedelta(days=30)) &
                (rate_snapshots["snapshot_date"] <= end_dt + pd.Timedelta(days=30)) &
                (rate_snapshots["availability_flag"] == 1)
            )
            available = rate_snapshots[wide_mask].copy()

        if available.empty:
            return None

        # Merge supplier reliability
        available = available.merge(
            supplier_rel[["supplier_id", "booking_failure_rate",
                          "preferred_supplier_flag"]],
            on="supplier_id", how="left"
        )

        # Score each option
        available["option_score"] = (
            (1 - available["booking_failure_rate"].fillna(0.2)) * 0.3 +
            available["preferred_supplier_flag"].fillna(0) * 0.25 +
            (available["cancellation_type"] == "Refundable").astype(float) * 0.25 +
            (1 - available["net_rate_inr"] / available["net_rate_inr"].max()) * 0.2
        )

        # Boost preferred supplier if specified
        if preferred_supplier:
            available.loc[
                available["supplier_id"] == preferred_supplier, "option_score"
            ] += 0.15

        # Aggregate by supplier (average score across dates)
        supplier_scores = (
            available.groupby("supplier_id")
            .agg(
                avg_score=("option_score", "mean"),
                avg_rate=("net_rate_inr", "mean"),
                available_dates=("snapshot_date", "nunique"),
                cancellation_type=("cancellation_type", "first"),
                cancel_penalty_pct=("cancel_penalty_pct", "mean"),
            )
            .reset_index()
            .sort_values("avg_score", ascending=False)
        )

        if supplier_scores.empty:
            return None

        best = supplier_scores.iloc[0]
        total_dates = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1

        return {
            "supplier_id": best["supplier_id"],
            "avg_rate": round(best["avg_rate"], 2),
            "available_dates": int(best["available_dates"]),
            "total_dates": total_dates,
            "coverage_pct": round(best["available_dates"] / total_dates * 100, 1),
            "cancellation_type": best["cancellation_type"],
            "cancel_penalty_pct": round(best["cancel_penalty_pct"], 3),
            "option_score": round(best["avg_score"], 4),
        }

    # ------------------------------------------------------------------
    # Execute Blocks
    # ------------------------------------------------------------------

    def execute_block(self, decision: dict) -> dict:
        """
        Execute a single blocking action:
          1. Check availability
          2. Confirm supplier terms
          3. Record in Demand_Block_Actions

        Returns the block record or error details.
        """
        prop_id = decision["property_id"]
        start = decision["block_start_date"]
        end = decision["block_end_date"]
        pref_supplier = decision.get("supplier_id", None)

        logger.info(f"Executing block for {prop_id} ({start} to {end}) ...")

        # Find best supplier option
        option = self.find_best_supplier_option(prop_id, start, end, pref_supplier)

        if option is None:
            logger.warning(f"  No availability found for {prop_id}. Block skipped.")
            self.failed_blocks.append({
                **decision,
                "failure_reason": "No supplier availability",
            })
            return {"status": "failed", "reason": "No availability"}

        if option["coverage_pct"] < 10:
            logger.warning(
                f"  Insufficient coverage ({option['coverage_pct']}%) for {prop_id}."
            )
            self.failed_blocks.append({
                **decision,
                "failure_reason": f"Low availability coverage: {option['coverage_pct']}%",
            })
            return {"status": "failed", "reason": "Insufficient coverage"}

        # Build block record
        block_id = f"BLK_{uuid.uuid4().hex[:8].upper()}"
        week_start = pd.to_datetime(start)

        block_record = {
            "block_id": block_id,
            "week_start": str(week_start.date()),
            "property_id": prop_id,
            "city": decision.get("city", ""),
            "block_start_date": start,
            "block_end_date": end,
            "rooms_blocked_per_night": decision["rooms_to_block"],
            "supplier_id": option["supplier_id"],
            "expected_revenue_uplift_inr": decision["expected_revenue_uplift"],
            "block_reason": decision["block_reason"],
        }

        # Check for existing block (same property + week_start + start_date)
        try:
            existing = read_table("demand_block_actions", engine=self.engine)
            if not existing.empty:
                dup = existing[
                    (existing["property_id"] == prop_id) &
                    (existing["week_start"].astype(str).str[:10] == str(week_start.date())) &
                    (existing["block_start_date"].astype(str).str[:10] == start)
                ]
                if not dup.empty:
                    logger.info(f"  Block already exists for {prop_id} week {week_start.date()}. Skipping.")
                    return {"status": "skipped", "reason": "Duplicate block"}
        except Exception as e:
            logger.warning(f"  Could not check existing blocks: {e}")

        # Insert into database
        try:
            block_df = pd.DataFrame([block_record])
            insert_dataframe(block_df, "demand_block_actions", engine=self.engine)
            logger.info(f"  Block {block_id} inserted into Demand_Block_Actions.")
        except Exception as e:
            logger.error(f"  Failed to insert block {block_id}: {e}")
            # Still track locally even if DB insert fails
            block_record["db_insert_error"] = str(e)

        self.executed_blocks.append(block_record)
        return {"status": "success", "block": block_record}

    def confirm_reservations(self) -> Dict[str, list]:
        """
        Batch execute all blocking decisions.
        Deduplicates by (property_id, block_start_date) before executing.

        Returns dict with 'executed' and 'failed' lists.
        """
        logger.info("=" * 60)
        logger.info("RESERVE AGENT: Confirming reservations")
        logger.info("=" * 60)

        if not self.decisions:
            logger.info("No blocking decisions to execute.")
            return {"executed": [], "failed": []}

        # Deduplicate: keep highest-scoring decision per property+start_date
        seen = {}
        for d in self.decisions:
            key = (d["property_id"], d["block_start_date"])
            if key not in seen or d.get("total_score", 0) > seen[key].get("total_score", 0):
                seen[key] = d
        unique_decisions = list(seen.values())
        if len(unique_decisions) < len(self.decisions):
            logger.info(f"  Deduplicated {len(self.decisions)} -> {len(unique_decisions)} decisions")

        for decision in unique_decisions:
            self.execute_block(decision)

        logger.info(
            f"Execution complete: {len(self.executed_blocks)} succeeded, "
            f"{len(self.failed_blocks)} failed."
        )

        return {
            "executed": self.executed_blocks,
            "failed": self.failed_blocks,
        }

    def get_execution_summary(self) -> dict:
        """Return summary of block execution results."""
        total_rooms = sum(b.get("rooms_blocked_per_night", 0) for b in self.executed_blocks)
        total_uplift = sum(b.get("expected_revenue_uplift_inr", 0) for b in self.executed_blocks)
        return {
            "total_blocks_executed": len(self.executed_blocks),
            "total_blocks_failed": len(self.failed_blocks),
            "total_rooms_blocked": total_rooms,
            "total_expected_uplift_inr": round(total_uplift, 2),
        }
