"""
OPTIMIZE AGENT - Smart Rebooking
==================================
Responsibilities:
  1. Evaluate rebooking opportunities (risk vs profit)
  2. Make rebook/skip decisions based on multi-criteria scoring
  3. Execute rebookings (cancel old, book new)
  4. Log all evaluations into Rebooking_Evaluations table
"""
import logging
import uuid
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from config.thresholds import (
    MIN_SAVINGS_INR,
    MIN_NET_PROFIT_INR,
    MIN_EQUIVALENCE_SCORE,
    MAX_RISK_SCORE,
    MIN_DAYS_BEFORE_DEADLINE,
    MAX_PENALTY_FRACTION,
    MIN_SUPPLIER_RELIABILITY,
    MIN_REBOOKING_CONFIDENCE,
    RISK_WEIGHT_CANCELLATION,
    RISK_WEIGHT_SUPPLIER,
    RISK_WEIGHT_EQUIVALENCE,
    RISK_WEIGHT_TIMING,
)
from utils.db_utils import get_sqlalchemy_engine, read_table, insert_dataframe

logger = logging.getLogger(__name__)


class OptimizeAgent:
    """
    Agent responsible for evaluating and executing smart rebooking
    decisions to recover margin on confirmed bookings.
    """

    def __init__(self, monitor_agent, engine=None):
        """
        Parameters
        ----------
        monitor_agent : MonitorAgent
            Provides rebooking candidates.
        engine : sqlalchemy.Engine, optional
        """
        self.monitor = monitor_agent
        self.engine = engine or get_sqlalchemy_engine()
        self.evaluations: List[dict] = []
        self.rebooked: List[dict] = []
        self.skipped: List[dict] = []

    # ------------------------------------------------------------------
    # Risk Scoring
    # ------------------------------------------------------------------

    def _cancellation_risk(self, cancel_type: str, penalty_pct: float,
                           days_to_deadline: int) -> float:
        """
        Score cancellation risk (0 = no risk, 100 = extreme risk).
        """
        # Type risk
        type_risk = {"Refundable": 0, "PartialRefund": 30, "NonRefundable": 100}
        base = type_risk.get(cancel_type, 50)

        # Penalty risk
        penalty_risk = min(penalty_pct * 100, 100)

        # Timing risk (fewer days = higher risk)
        if days_to_deadline <= 0:
            timing = 100
        elif days_to_deadline <= 2:
            timing = 70
        elif days_to_deadline <= 7:
            timing = 30
        else:
            timing = 10

        return (base * 0.4 + penalty_risk * 0.3 + timing * 0.3)

    def _supplier_risk(self, new_supplier_id: str) -> float:
        """
        Score supplier risk based on reliability data (0 = safe, 100 = risky).
        """
        suppliers = read_table("supplier_reliability", engine=self.engine)
        sup = suppliers[suppliers["supplier_id"] == new_supplier_id]

        if sup.empty:
            return 50.0  # Unknown supplier = moderate risk

        row = sup.iloc[0]
        failure = row.get("booking_failure_rate", 0.1)
        cancel = row.get("supplier_cancellation_rate", 0.1)
        dispute = row.get("dispute_rate", 0.05)
        preferred = row.get("preferred_supplier_flag", 0)

        risk = (failure * 40 + cancel * 30 + dispute * 20) * 100
        if preferred:
            risk *= 0.6  # Preferred suppliers get a discount

        return min(risk, 100)

    def _equivalence_risk(self, equivalence_score: float) -> float:
        """
        Score room equivalence risk (0 = perfect match, 100 = poor match).
        """
        return max(0, (1 - equivalence_score) * 100)

    def _timing_risk(self, days_to_deadline: int) -> float:
        """
        Score timing risk based on proximity to cancellation deadline.
        """
        if days_to_deadline <= 1:
            return 90
        elif days_to_deadline <= 3:
            return 60
        elif days_to_deadline <= 7:
            return 30
        elif days_to_deadline <= 14:
            return 15
        return 5

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_rebooking_opportunity(self, candidate: dict) -> dict:
        """
        Evaluate a rebooking candidate with risk and profit scoring.

        Returns evaluation dict with decision.
        """
        old_cost = candidate["old_cost"]
        new_cost = candidate["new_cost"]
        savings = candidate["savings"]
        penalty = candidate.get("estimated_penalty", 0)
        eq_score = candidate.get("equivalence_score", 0.9)
        cancel_type = candidate.get("cancellation_type", "Refundable")
        penalty_pct = candidate.get("cancel_penalty_pct", 0)
        days_to_deadline = candidate.get("days_to_deadline", 30)
        new_supplier = candidate.get("new_supplier", "")

        # Calculate risk score (weighted)
        cr = self._cancellation_risk(cancel_type, penalty_pct, days_to_deadline)
        sr = self._supplier_risk(new_supplier)
        er = self._equivalence_risk(eq_score)
        tr = self._timing_risk(days_to_deadline)

        risk_score = (
            RISK_WEIGHT_CANCELLATION * cr +
            RISK_WEIGHT_SUPPLIER * sr +
            RISK_WEIGHT_EQUIVALENCE * er +
            RISK_WEIGHT_TIMING * tr
        )

        # Calculate profit score
        net_profit = savings - penalty
        profit_score = (net_profit / old_cost * 100) if old_cost > 0 else 0

        # Calculate confidence
        confidence = np.clip(
            1.0 - risk_score / 100 * 0.5 + profit_score / 100 * 0.3,
            0.0, 1.0
        )

        # Decision logic
        decision = "Skip"
        decision_reasons = []

        if savings < MIN_SAVINGS_INR:
            decision_reasons.append(f"Savings {savings:.0f} < min {MIN_SAVINGS_INR:.0f}")
        if net_profit < MIN_NET_PROFIT_INR:
            decision_reasons.append(f"Net profit {net_profit:.0f} < min {MIN_NET_PROFIT_INR:.0f}")
        if eq_score < MIN_EQUIVALENCE_SCORE:
            decision_reasons.append(f"Equivalence {eq_score:.2f} < min {MIN_EQUIVALENCE_SCORE:.2f}")
        if risk_score > MAX_RISK_SCORE:
            decision_reasons.append(f"Risk {risk_score:.1f} > max {MAX_RISK_SCORE:.1f}")
        if days_to_deadline < MIN_DAYS_BEFORE_DEADLINE:
            decision_reasons.append(f"Days to deadline {days_to_deadline} < min {MIN_DAYS_BEFORE_DEADLINE}")
        if penalty_pct > MAX_PENALTY_FRACTION:
            decision_reasons.append(f"Penalty {penalty_pct:.1%} > max {MAX_PENALTY_FRACTION:.1%}")
        if confidence < MIN_REBOOKING_CONFIDENCE:
            decision_reasons.append(f"Confidence {confidence:.2f} < min {MIN_REBOOKING_CONFIDENCE:.2f}")

        if not decision_reasons:
            decision = "Rebook"

        evaluation = {
            **candidate,
            "estimated_penalty_inr": round(penalty, 2),
            "net_profit": round(net_profit, 2),
            "risk_score": round(risk_score, 2),
            "profit_score": round(profit_score, 2),
            "confidence": round(confidence, 3),
            "decision": decision,
            "decision_reasons": "; ".join(decision_reasons) if decision_reasons else "All criteria met",
            "risk_breakdown": {
                "cancellation_risk": round(cr, 2),
                "supplier_risk": round(sr, 2),
                "equivalence_risk": round(er, 2),
                "timing_risk": round(tr, 2),
            },
        }

        return evaluation

    def make_rebooking_decision(self, evaluation: dict) -> str:
        """Return the decision from an evaluation (Rebook or Skip)."""
        return evaluation.get("decision", "Skip")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_rebooking(self, evaluation: dict, eval_week: str = None) -> dict:
        """
        Log the rebooking evaluation into Rebooking_Evaluations table.
        If decision is 'Rebook', the booking is marked for rebooking.

        Parameters
        ----------
        evaluation : dict
            Output from evaluate_rebooking_opportunity.
        eval_week : str, optional
            Week start date for grouping.

        Returns
        -------
        dict with execution status.
        """
        eval_id = f"RE_{uuid.uuid4().hex[:8].upper()}"
        if eval_week is None:
            eval_week = str(pd.Timestamp.now().normalize().date())

        record = {
            "rebooking_eval_id": eval_id,
            "booking_id": evaluation.get("booking_id", ""),
            "evaluation_date": str(pd.Timestamp.now().date()),
            "property_id": evaluation.get("property_id", ""),
            "city": evaluation.get("city", ""),
            "room_type": evaluation.get("room_type", ""),
            "meal_plan": evaluation.get("meal_plan", ""),
            "old_supplier": evaluation.get("old_supplier", ""),
            "new_supplier": evaluation.get("new_supplier", ""),
            "old_cost_inr": evaluation.get("old_cost", 0),
            "new_cost_inr": evaluation.get("new_cost", 0),
            "savings_inr": evaluation.get("savings", 0),
            "estimated_penalty_inr": evaluation.get("estimated_penalty_inr", 0),
            "risk_score": evaluation.get("risk_score", 0),
            "profit_score": evaluation.get("profit_score", 0),
            "confidence": evaluation.get("confidence", 0),
            "decision": evaluation.get("decision", "Skip"),
            "eval_week": eval_week,
        }

        # Check for existing evaluation (same booking + supplier + week)
        try:
            existing = read_table("rebooking_evaluations", engine=self.engine)
            if not existing.empty:
                dup = existing[
                    (existing["booking_id"] == record["booking_id"]) &
                    (existing["new_supplier"] == record["new_supplier"]) &
                    (existing["eval_week"].astype(str).str[:10] == str(eval_week)[:10])
                ]
                if not dup.empty:
                    logger.info(f"  Evaluation already exists for {record['booking_id']} + {record['new_supplier']} week {eval_week}. Skipping.")
                    record["rebooking_eval_id"] = dup.iloc[0]["rebooking_eval_id"]
                    self.evaluations.append(record)
                    if record["decision"] == "Rebook":
                        self.rebooked.append(record)
                    else:
                        self.skipped.append(record)
                    return record
        except Exception as e:
            logger.warning(f"  Could not check existing evaluations: {e}")

        # Insert into database
        try:
            record_df = pd.DataFrame([record])
            insert_dataframe(record_df, "rebooking_evaluations", engine=self.engine)
            logger.info(f"  Logged rebooking evaluation {eval_id} -> {record['decision']}")
        except Exception as e:
            logger.error(f"  Failed to log evaluation {eval_id}: {e}")
            record["db_insert_error"] = str(e)

        self.evaluations.append(record)
        if record["decision"] == "Rebook":
            self.rebooked.append(record)
        else:
            self.skipped.append(record)

        return record

    # ------------------------------------------------------------------
    # Batch Processing
    # ------------------------------------------------------------------

    def batch_process_rebookings(self, eval_week: str = None) -> Dict[str, list]:
        """
        Evaluate and process all rebooking candidates from MonitorAgent.

        Returns dict with 'rebooked', 'skipped', and 'evaluations' lists.
        """
        logger.info("=" * 60)
        logger.info("OPTIMIZE AGENT: Processing rebooking candidates")
        logger.info("=" * 60)

        candidates = self.monitor.get_rebooking_candidates()

        if not candidates:
            logger.info("No rebooking candidates to process.")
            return {"rebooked": [], "skipped": [], "evaluations": []}

        for candidate in candidates:
            evaluation = self.evaluate_rebooking_opportunity(candidate)
            self.execute_rebooking(evaluation, eval_week)

        logger.info(
            f"Rebooking complete: {len(self.rebooked)} rebooked, "
            f"{len(self.skipped)} skipped out of {len(self.evaluations)} evaluated."
        )

        return {
            "rebooked": self.rebooked,
            "skipped": self.skipped,
            "evaluations": self.evaluations,
        }

    def get_optimization_summary(self) -> dict:
        """Return summary of rebooking optimization results."""
        total_savings = sum(r.get("savings_inr", 0) for r in self.rebooked)
        total_penalty = sum(r.get("estimated_penalty_inr", 0) for r in self.rebooked)
        avg_confidence = (
            np.mean([r.get("confidence", 0) for r in self.rebooked])
            if self.rebooked else 0
        )

        return {
            "total_evaluations": len(self.evaluations),
            "total_rebooked": len(self.rebooked),
            "total_skipped": len(self.skipped),
            "total_savings_inr": round(total_savings, 2),
            "total_penalties_inr": round(total_penalty, 2),
            "net_margin_recovered_inr": round(total_savings - total_penalty, 2),
            "avg_confidence": round(avg_confidence, 3),
            "rebook_rate_pct": round(
                len(self.rebooked) / max(len(self.evaluations), 1) * 100, 1
            ),
        }
