"""
MONITOR AGENT - Track Bookings for Rebooking Opportunities
============================================================
Responsibilities:
  1. Query confirmed bookings eligible for monitoring
  2. Search Rate_Snapshots for cheaper rates on same/equivalent rooms
  3. Verify cancellation safety windows
  4. Surface rebooking candidates for the OptimizeAgent
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from utils.db_utils import get_sqlalchemy_engine, read_table

logger = logging.getLogger(__name__)


class MonitorAgent:
    """
    Agent that continuously monitors confirmed bookings against
    current market rates to identify rebooking opportunities.
    """

    def __init__(self, engine=None):
        self.engine = engine or get_sqlalchemy_engine()
        self.active_bookings: pd.DataFrame = pd.DataFrame()
        self.rebooking_candidates: List[dict] = []
        # Cache table reads for performance
        self._rate_snapshots = None
        self._room_mapping = None

    def _get_rate_snapshots(self) -> pd.DataFrame:
        if self._rate_snapshots is None:
            self._rate_snapshots = read_table("rate_snapshots", engine=self.engine)
            self._rate_snapshots["snapshot_date"] = pd.to_datetime(self._rate_snapshots["snapshot_date"])
        return self._rate_snapshots

    def _get_room_mapping(self) -> pd.DataFrame:
        if self._room_mapping is None:
            self._room_mapping = read_table("room_mapping", engine=self.engine)
        return self._room_mapping

    # ------------------------------------------------------------------
    # 1. Get Active Bookings
    # ------------------------------------------------------------------

    def get_active_bookings(self, reference_date: str = None) -> pd.DataFrame:
        """
        Fetch confirmed bookings that are:
          - status = 'Confirmed'
          - check-in date is in the future (relative to reference_date)
          - cancellation deadline has not passed

        Parameters
        ----------
        reference_date : str, optional
            Treat this date as 'today'. Defaults to the max booked_date in data.
        """
        logger.info("MONITOR AGENT: Fetching active bookings ...")

        bookings = read_table("confirmed_bookings", engine=self.engine)
        bookings["booked_date"] = pd.to_datetime(bookings["booked_date"])
        bookings["stay_checkin_date"] = pd.to_datetime(bookings["stay_checkin_date"])
        bookings["stay_checkout_date"] = pd.to_datetime(bookings["stay_checkout_date"])
        bookings["cancel_deadline_date"] = pd.to_datetime(bookings["cancel_deadline_date"])

        if reference_date is None:
            reference_date = bookings["booked_date"].max()
        else:
            reference_date = pd.to_datetime(reference_date)

        # Filter eligible bookings
        mask = (
            (bookings["booking_status"] == "Confirmed") &
            (bookings["cancellation_type"] != "NonRefundable") &
            (bookings["stay_checkin_date"] > reference_date) &
            (bookings["cancel_deadline_date"] >= reference_date)
        )
        self.active_bookings = bookings[mask].copy()

        logger.info(f"  Found {len(self.active_bookings)} active bookings eligible for monitoring.")
        return self.active_bookings

    # ------------------------------------------------------------------
    # 2. Find Cheaper Rates
    # ------------------------------------------------------------------

    def find_cheaper_rates(self, booking: pd.Series) -> List[dict]:
        """
        For a single booking, search Rate_Snapshots for cheaper options
        on the same property (same or equivalent room type).

        Parameters
        ----------
        booking : pd.Series
            A single row from active_bookings.

        Returns
        -------
        List of dicts, each representing a cheaper rate option.
        """
        rate_snapshots = self._get_rate_snapshots()
        room_mapping = self._get_room_mapping()

        property_id = booking["property_id"]
        room_type = booking["standard_room_type"]
        meal_plan = booking["meal_plan"]
        booked_rate = booking["booked_net_rate_inr"]
        old_supplier = booking["supplier_id_booked"]

        # Find equivalent room types via Room_Mapping
        equiv = room_mapping[
            (room_mapping["property_id"] == property_id) &
            (room_mapping["upgrade_only_flag"] == 0)
        ]
        equivalent_rooms = equiv[
            (equiv["standard_room_type"] == room_type) &
            (equiv["equivalence_score"] >= 0.80)
        ]
        acceptable_room_types = equivalent_rooms["standard_room_type"].unique().tolist()
        if room_type not in acceptable_room_types:
            acceptable_room_types.append(room_type)

        # Search for cheaper rates (latest snapshot on or before check-in date)
        checkin_date = pd.to_datetime(booking["stay_checkin_date"])
        prop_snapshots = rate_snapshots[
            (rate_snapshots["property_id"] == property_id) &
            (rate_snapshots["snapshot_date"] <= checkin_date)
        ]
        latest_snapshot_date = prop_snapshots["snapshot_date"].max() if not prop_snapshots.empty else None
        if latest_snapshot_date is None:
            return []
        mask = (
            (rate_snapshots["property_id"] == property_id) &
            (rate_snapshots["standard_room_type"].isin(acceptable_room_types)) &
            (rate_snapshots["meal_plan"] == meal_plan) &
            (rate_snapshots["availability_flag"] == 1) &
            (rate_snapshots["net_rate_inr"] < booked_rate) &
            (rate_snapshots["snapshot_date"] == latest_snapshot_date) &
            (rate_snapshots["supplier_id"] != old_supplier) &
            (rate_snapshots["cancellation_type"] != "NonRefundable")
        )
        cheaper = rate_snapshots[mask].copy()

        if cheaper.empty:
            return []

        # Enrich with equivalence scores
        options = []
        for _, snap in cheaper.iterrows():
            # Get equivalence score
            eq_match = equiv[
                (equiv["supplier_id"] == snap["supplier_id"]) &
                (equiv["standard_room_type"] == snap["standard_room_type"])
            ]
            eq_score = eq_match["equivalence_score"].max() if not eq_match.empty else 0.9

            savings = booked_rate - snap["net_rate_inr"]
            options.append({
                "booking_id": booking["booking_id"],
                "property_id": property_id,
                "city": booking["city"],
                "room_type": snap["standard_room_type"],
                "meal_plan": snap["meal_plan"],
                "old_supplier": old_supplier,
                "new_supplier": snap["supplier_id"],
                "old_cost": booked_rate,
                "new_cost": snap["net_rate_inr"],
                "savings": round(savings, 2),
                "equivalence_score": round(eq_score, 3),
                "cancellation_type": snap["cancellation_type"],
                "cancel_penalty_pct": snap["cancel_penalty_pct"],
                "cancel_deadline_days": snap.get("cancel_deadline_days", 0),
            })

        # Sort by savings (descending)
        options.sort(key=lambda x: x["savings"], reverse=True)
        return options

    # ------------------------------------------------------------------
    # 3. Cancellation Safety Check
    # ------------------------------------------------------------------

    def check_cancellation_safety(self, booking: pd.Series,
                                  reference_date: str = None) -> dict:
        """
        Verify whether rebooking is safe within the cancellation window.

        Returns dict with safety assessment.
        """
        if reference_date is None:
            reference_date = pd.Timestamp.now().normalize()
        else:
            reference_date = pd.to_datetime(reference_date)

        deadline = booking["cancel_deadline_date"]
        cancel_type = booking["cancellation_type"]
        penalty_pct = booking["cancel_penalty_pct"]
        booked_rate = booking["booked_net_rate_inr"]

        days_to_deadline = (deadline - reference_date).days

        # Calculate penalty
        if cancel_type == "Refundable":
            estimated_penalty = 0.0
        elif cancel_type == "PartialRefund":
            if days_to_deadline <= 0:
                estimated_penalty = booked_rate * penalty_pct
            else:
                estimated_penalty = booked_rate * penalty_pct * max(0, 1 - days_to_deadline / 14)
        else:  # NonRefundable
            estimated_penalty = booked_rate * penalty_pct

        is_safe = (
            days_to_deadline >= 2 and
            cancel_type != "NonRefundable"
        )

        return {
            "booking_id": booking["booking_id"],
            "days_to_deadline": days_to_deadline,
            "cancellation_type": cancel_type,
            "estimated_penalty": round(estimated_penalty, 2),
            "is_safe": is_safe,
            "safety_reason": (
                "Safe to cancel" if is_safe else
                f"Unsafe: {'Past deadline' if days_to_deadline < 2 else 'NonRefundable'}"
            ),
        }

    # ------------------------------------------------------------------
    # 4. Batch Processing
    # ------------------------------------------------------------------

    def find_cheaper_rates_for_all(self, reference_date: str = None) -> List[dict]:
        """
        Scan all active bookings for rebooking opportunities.

        Returns list of candidate dicts with booking + cheaper option details.
        """
        logger.info("=" * 60)
        logger.info("MONITOR AGENT: Scanning for rebooking opportunities")
        logger.info("=" * 60)

        if self.active_bookings.empty:
            self.get_active_bookings(reference_date)

        if self.active_bookings.empty:
            logger.info("No active bookings to monitor.")
            return []

        candidates = []
        scanned = 0
        found = 0

        for _, booking in self.active_bookings.iterrows():
            scanned += 1

            # Check cancellation safety first
            safety = self.check_cancellation_safety(booking, reference_date)
            if not safety["is_safe"]:
                continue

            # Find cheaper rates
            options = self.find_cheaper_rates(booking)
            if not options:
                continue

            # Attach safety info to each option
            for opt in options:
                opt["days_to_deadline"] = safety["days_to_deadline"]
                opt["estimated_penalty"] = safety["estimated_penalty"]
                opt["cancellation_safe"] = True
                candidates.append(opt)
                found += 1

            if scanned % 500 == 0:
                logger.info(f"  Scanned {scanned}/{len(self.active_bookings)} bookings, found {found} candidates ...")

        self.rebooking_candidates = candidates
        logger.info(f"  Total: scanned {scanned} bookings, found {found} rebooking candidates.")
        return candidates

    def get_rebooking_candidates(self) -> List[dict]:
        """Return the identified rebooking candidates."""
        return self.rebooking_candidates
