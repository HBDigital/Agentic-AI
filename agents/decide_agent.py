"""
DECIDE AGENT - Demand Blocking Strategy
=========================================
Responsibilities:
  1. Consume predictions from PredictAgent
  2. Calculate composite blocking scores per property
  3. Rank and select properties for inventory blocking
  4. Determine optimal rooms to block and preferred supplier
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from config.thresholds import (
    DEMAND_SPIKE_THRESHOLD,
    SELLOUT_PROBABILITY_THRESHOLD,
    REVENUE_SCORE_THRESHOLD,
    SUPPLIER_FAILURE_RATE_THRESHOLD,
    PRICE_INFLATION_THRESHOLD,
    MIN_BLOCKING_SCORE,
    MAX_BLOCK_FRACTION,
    MIN_ROOMS_TO_BLOCK,
    BLOCKING_WEIGHT_DEMAND,
    BLOCKING_WEIGHT_SELLOUT,
    BLOCKING_WEIGHT_REVENUE,
    BLOCKING_WEIGHT_SUPPLIER,
    BLOCKING_WEIGHT_PRICE,
)

logger = logging.getLogger(__name__)


class DecideAgent:
    """
    Agent responsible for deciding which properties to block,
    how many rooms, and with which supplier.
    """

    def __init__(self, predict_agent, sense_agent):
        """
        Parameters
        ----------
        predict_agent : PredictAgent
            Provides demand, sell-out, and price predictions.
        sense_agent : SenseAgent
            Provides raw data for supplier info, property master, etc.
        """
        self.predict = predict_agent
        self.raw_data = sense_agent.get_raw_data()
        self.blocking_decisions: List[dict] = []

    # ------------------------------------------------------------------
    # Score Components
    # ------------------------------------------------------------------

    def _demand_score(self, predicted_demand: float) -> float:
        """Normalize demand prediction to [0, 1] score."""
        if predicted_demand <= 1.0:
            return 0.0
        # Gradual scaling: demand of 1.1 -> 0.33, 1.2 -> 0.67, 1.3+ -> 1.0
        return min((predicted_demand - 1.0) / (DEMAND_SPIKE_THRESHOLD - 1.0), 1.0)

    def _sellout_score(self, sellout_prob: float) -> float:
        """Normalize sell-out probability to [0, 1] score.
        Uses sigmoid-like scaling to amplify even small probabilities."""
        # Amplify: prob of 0.01 -> 0.18, 0.05 -> 0.53, 0.10 -> 0.77, 0.5 -> 1.0
        scaled = min(sellout_prob / SELLOUT_PROBABILITY_THRESHOLD, 1.0)
        return scaled ** 0.5  # square root to amplify small values

    def _revenue_score(self, base_adr: float, popularity: float) -> float:
        """Normalize revenue potential to [0, 1] score."""
        raw = base_adr * popularity
        return min(raw / (REVENUE_SCORE_THRESHOLD * 2), 1.0)

    def _supplier_score(self, failure_rate: float, preferred: int) -> float:
        """Score supplier reliability (higher is better)."""
        reliability = 1.0 - failure_rate
        bonus = 0.2 if preferred else 0.0
        return min(reliability + bonus, 1.0)

    def _price_score(self, current_rate: float, predicted_rate: float) -> float:
        """Score expected price inflation (higher inflation = higher score)."""
        if current_rate <= 0:
            return 0.0
        inflation = (predicted_rate - current_rate) / current_rate
        if inflation <= 0:
            return 0.0
        return min(inflation / (PRICE_INFLATION_THRESHOLD * 2), 1.0)

    # ------------------------------------------------------------------
    # Composite Blocking Score
    # ------------------------------------------------------------------

    def calculate_blocking_score(self, demand_val: float, sellout_val: float,
                                 base_adr: float, popularity: float,
                                 failure_rate: float, preferred: int,
                                 current_rate: float, predicted_rate: float) -> dict:
        """
        Calculate composite blocking score with breakdown.

        Returns
        -------
        dict with total_score and component scores.
        """
        ds = self._demand_score(demand_val)
        ss = self._sellout_score(sellout_val)
        rs = self._revenue_score(base_adr, popularity)
        sups = self._supplier_score(failure_rate, preferred)
        ps = self._price_score(current_rate, predicted_rate)

        total = (
            BLOCKING_WEIGHT_DEMAND * ds +
            BLOCKING_WEIGHT_SELLOUT * ss +
            BLOCKING_WEIGHT_REVENUE * rs +
            BLOCKING_WEIGHT_SUPPLIER * sups +
            BLOCKING_WEIGHT_PRICE * ps
        )

        return {
            "total_score": round(total, 4),
            "demand_score": round(ds, 4),
            "sellout_score": round(ss, 4),
            "revenue_score": round(rs, 4),
            "supplier_score": round(sups, 4),
            "price_score": round(ps, 4),
        }

    # ------------------------------------------------------------------
    # Property Selection
    # ------------------------------------------------------------------

    def select_properties_to_block(self, week_start: str = None) -> List[dict]:
        """
        Rank all properties by blocking score and select those
        above the threshold.

        Parameters
        ----------
        week_start : str, optional
            Week start date for the blocking period.

        Returns
        -------
        List of blocking decision dicts.
        """
        logger.info("=" * 60)
        logger.info("DECIDE AGENT: Selecting properties to block")
        logger.info("=" * 60)

        # Gather predictions
        demand_preds = self.predict.predict_demand()
        sellout_preds = self.predict.predict_sellout()
        price_preds = self.predict.predict_prices()

        # Property master for base attributes
        pm = self.raw_data["property_master"].copy()
        suppliers = self.raw_data["supplier_reliability"].copy()

        # Aggregate demand predictions per city
        city_demand = (
            demand_preds.groupby("city")
            .agg(predicted_demand=("predicted_demand_index", "mean"),
                 demand_confidence=("confidence", "mean"))
            .reset_index()
        )

        # Aggregate sellout predictions per property
        prop_sellout = (
            sellout_preds.groupby(["property_id", "city"])
            .agg(sellout_prob=("sellout_probability", "max"),
                 booking_surge=("expected_booking_surge", "max"))
            .reset_index()
        )

        # Aggregate price predictions per property
        if not price_preds.empty:
            prop_price = (
                price_preds.groupby("property_id")
                .agg(current_rate=("current_rate", "mean"),
                     predicted_rate=("predicted_rate_7d", "mean"))
                .reset_index()
            )
        else:
            prop_price = pd.DataFrame(columns=["property_id", "current_rate", "predicted_rate"])

        # Get best supplier per property
        best_suppliers = self._get_best_suppliers()

        # Build candidate list
        candidates = pm.merge(prop_sellout, on="property_id", how="left", suffixes=("", "_so"))
        candidates = candidates.merge(city_demand, on="city", how="left")
        candidates = candidates.merge(prop_price, on="property_id", how="left")
        candidates = candidates.merge(best_suppliers, on="property_id", how="left")

        # Fill missing values
        candidates["predicted_demand"] = candidates["predicted_demand"].fillna(1.0)
        candidates["sellout_prob"] = candidates["sellout_prob"].fillna(0.0)
        candidates["current_rate"] = candidates["current_rate"].fillna(
            candidates["base_adr_inr"]
        )
        candidates["predicted_rate"] = candidates["predicted_rate"].fillna(
            candidates["current_rate"]
        )
        candidates["best_failure_rate"] = candidates["best_failure_rate"].fillna(0.1)
        candidates["best_preferred"] = candidates["best_preferred"].fillna(0).astype(int)

        # Calculate blocking scores
        decisions = []
        for _, row in candidates.iterrows():
            scores = self.calculate_blocking_score(
                demand_val=row["predicted_demand"],
                sellout_val=row["sellout_prob"],
                base_adr=row["base_adr_inr"],
                popularity=row["popularity_index"],
                failure_rate=row["best_failure_rate"],
                preferred=row["best_preferred"],
                current_rate=row["current_rate"],
                predicted_rate=row["predicted_rate"],
            )

            if scores["total_score"] < MIN_BLOCKING_SCORE:
                continue

            # Require at least one strong demand signal to justify blocking
            has_sellout_signal = row["sellout_prob"] >= SELLOUT_PROBABILITY_THRESHOLD
            has_demand_signal = row["predicted_demand"] >= DEMAND_SPIKE_THRESHOLD
            if not has_sellout_signal and not has_demand_signal:
                continue

            # Block dates
            if week_start:
                block_start = pd.to_datetime(week_start)
            else:
                block_start = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
            block_end = block_start + pd.Timedelta(days=6)

            # Check events - match by month/day (ignore year) so annual events apply
            events = self.raw_data["events_calendar"].copy()
            events["start_date"] = pd.to_datetime(events["start_date"])
            events["end_date"] = pd.to_datetime(events["end_date"])
            block_year = block_start.year

            def _shift_year(d, yr):
                """Shift date to target year, handling leap-year edge cases."""
                try:
                    return d.replace(year=yr)
                except ValueError:
                    # e.g. Feb 29 in a non-leap year -> Feb 28
                    import calendar
                    last_day = calendar.monthrange(yr, d.month)[1]
                    return d.replace(year=yr, day=min(d.day, last_day))

            events["start_date_adj"] = events["start_date"].apply(
                lambda d: _shift_year(d, block_year)
            )
            events["end_date_adj"] = events["end_date"].apply(
                lambda d: _shift_year(d, block_year)
            )
            city_events = events[
                (events["city"] == row.get("city", "")) &
                (events["start_date_adj"] <= block_end) &
                (events["end_date_adj"] >= block_start)
            ]
            event_intensity = float(city_events["demand_intensity"].max()) if not city_events.empty else 0.0

            # Determine rooms to block (boosted by event intensity)
            rooms = self._determine_block_quantity(row, event_intensity)

            # Build block reason
            reasons = []
            if row["sellout_prob"] >= SELLOUT_PROBABILITY_THRESHOLD:
                reasons.append(f"High sellout risk ({row['sellout_prob']:.2f})")
            if row["predicted_demand"] >= DEMAND_SPIKE_THRESHOLD:
                reasons.append(f"Demand spike ({row['predicted_demand']:.2f})")
            if scores["price_score"] > 0.3:
                reasons.append("Price inflation expected")
            if event_intensity > 0:
                reasons.append(f"Event activity in city (intensity {event_intensity:.0f})")

            if not reasons:
                reasons.append(f"Composite score {scores['total_score']:.2f}")

            # Calculate expected revenue uplift (boosted by event intensity)
            uplift = self._estimate_revenue_uplift(row, rooms, event_intensity)

            decision = {
                "property_id": row["property_id"],
                "property_name": row.get("property_name", ""),
                "city": row.get("city", row.get("city_so", "")),
                "block_start_date": str(block_start.date()),
                "block_end_date": str(block_end.date()),
                "rooms_to_block": rooms,
                "supplier_id": row.get("best_supplier_id", ""),
                "expected_revenue_uplift": round(uplift, 2),
                "block_reason": " + ".join(reasons),
                "confidence_score": round(
                    scores["total_score"] * row.get("demand_confidence", 0.85), 3
                ),
                **scores,
            }
            decisions.append(decision)

        # Sort by total_score descending
        decisions.sort(key=lambda x: x["total_score"], reverse=True)

        self.blocking_decisions = decisions
        logger.info(f"Selected {len(decisions)} properties for blocking")
        return decisions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _determine_block_quantity(self, row: pd.Series, event_intensity: float = 0.0) -> int:
        """Calculate optimal number of rooms to block.
        Event intensity (1-3) boosts the fraction of rooms blocked."""
        base_inventory = row.get("base_inventory_rooms", 20)
        sellout_prob = row.get("sellout_prob", 0.0)
        demand = row.get("predicted_demand", 1.0)

        # Scale rooms by sellout probability and demand
        fraction = sellout_prob * 0.4 + (demand - 1.0) * 0.2

        # Boost by event intensity: intensity 1 -> +10%, 2 -> +20%, 3 -> +30%
        if event_intensity > 0:
            fraction += event_intensity * 0.10

        fraction = min(fraction, MAX_BLOCK_FRACTION)
        rooms = max(int(np.ceil(base_inventory * fraction)), MIN_ROOMS_TO_BLOCK)
        rooms = min(rooms, int(base_inventory * MAX_BLOCK_FRACTION))
        return rooms

    def _estimate_revenue_uplift(self, row: pd.Series, rooms: int, event_intensity: float = 0.0) -> float:
        """Estimate expected revenue uplift from blocking.
        Event intensity boosts the demand premium."""
        adr = row.get("base_adr_inr", 5000)
        demand_mult = row.get("predicted_demand", 1.0)
        sellout_prob = row.get("sellout_prob", 0.5)

        # Revenue uplift = rooms x nights x ADR x demand premium x sellout capture probability
        nights = 7
        premium = max(demand_mult - 1.0, 0.1) * 0.5

        # Event boost: intensity 1 -> +15%, 2 -> +30%, 3 -> +45% premium
        if event_intensity > 0:
            premium *= (1.0 + event_intensity * 0.15)

        uplift = rooms * nights * adr * premium * sellout_prob
        return uplift

    def _get_best_suppliers(self) -> pd.DataFrame:
        """
        For each property, find the best supplier based on room mapping
        and supplier reliability.
        """
        rm = self.raw_data["room_mapping"].copy()
        sup = self.raw_data["supplier_reliability"].copy()

        # Merge supplier info
        merged = rm.merge(sup, on="supplier_id", how="left")

        # Score: prefer preferred suppliers with low failure rates and high equivalence
        merged["sup_score"] = (
            merged["equivalence_score"] * 0.4 +
            (1 - merged["booking_failure_rate"].fillna(0.2)) * 0.3 +
            merged["preferred_supplier_flag"].fillna(0) * 0.3
        )

        # Pick best supplier per property
        best = (
            merged.sort_values("sup_score", ascending=False)
            .drop_duplicates("property_id")
            [["property_id", "supplier_id", "booking_failure_rate", "preferred_supplier_flag"]]
            .rename(columns={
                "supplier_id": "best_supplier_id",
                "booking_failure_rate": "best_failure_rate",
                "preferred_supplier_flag": "best_preferred",
            })
        )
        return best

    def get_blocking_decisions(self) -> List[dict]:
        """Return the latest blocking decisions."""
        return self.blocking_decisions
