"""
MAIN ORCHESTRATOR
==================
Runs the full 7-agent pipeline for Demand Blocking & Smart Rebooking.

Usage:
    python main.py                          # Run with defaults
    python main.py --week-start 2026-02-16  # Specify week start
    python main.py --train                  # Force model retraining
    python main.py --report-only            # Generate report only
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("orchestrator")

# ---------------------------------------------------------------------------
# Project root (used for model paths)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.db_utils import get_sqlalchemy_engine, test_connection
from agents.sense_agent import SenseAgent
from agents.predict_agent import PredictAgent
from agents.decide_agent import DecideAgent
from agents.reserve_agent import ReserveAgent
from agents.monitor_agent import MonitorAgent
from agents.optimize_agent import OptimizeAgent
from agents.report_agent import ReportAgent


def run_weekly_pipeline(week_start: str, force_train: bool = False,
                        report_only: bool = False):
    """
    Execute the full 7-step agentic AI pipeline.

    Parameters
    ----------
    week_start : str
        Monday of the target week (YYYY-MM-DD).
    force_train : bool
        If True, retrain models even if saved models exist.
    report_only : bool
        If True, skip execution and only generate the report.
    """
    pipeline_start = time.time()
    logger.info("=" * 70)
    logger.info("   AGENTIC AI PIPELINE -- DEMAND BLOCKING & SMART REBOOKING")
    logger.info(f"   Week Start: {week_start}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Database connection
    # ------------------------------------------------------------------
    engine = get_sqlalchemy_engine()
    if not test_connection(engine):
        logger.error("[ERROR] Database connection failed. Aborting pipeline.")
        return

    if report_only:
        logger.info("Report-only mode. Skipping Steps 1-6.")
        report = ReportAgent(engine)
        report_path = report.create_executive_report(week_start, output_dir="reports")
        logger.info(f"[OK] Report generated: {report_path}")
        return

    # ------------------------------------------------------------------
    # STEP 1 -- SENSE AGENT
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 70)
    logger.info("  STEP 1: SENSE AGENT -- Data Ingestion & Feature Engineering")
    logger.info("-" * 70)

    sense = SenseAgent(engine)
    sense.ingest_data()
    sense.validate_data()
    sense.handle_missing_data()
    sense.engineer_features()

    state = sense.get_current_state()
    logger.info(f"  Current state: {state['total_properties']} properties, "
                f"avg occupancy {state['avg_occupancy']:.1%}, "
                f"{state['soldout_count']} sold out")

    # ------------------------------------------------------------------
    # STEP 2 -- PREDICT AGENT
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 70)
    logger.info("  STEP 2: PREDICT AGENT -- ML Model Training & Prediction")
    logger.info("-" * 70)

    predict = PredictAgent(sense, project_root=PROJECT_ROOT)

    # Check for saved models
    models_exist = all(
        os.path.exists(os.path.join(PROJECT_ROOT, p))
        for p in ["models/demand_model.pkl", "models/sellout_model.pkl", "models/price_model.pkl"]
    )

    if force_train or not models_exist:
        logger.info("Training all ML models ...")
        metrics = predict.train_all_models()
        predict.save_models()

        logger.info("\n--- MODEL PERFORMANCE SUMMARY ---")
        for model_name, m in metrics.items():
            logger.info(f"  {model_name}: {m}")
    else:
        logger.info("Loading pre-trained models ...")
        predict.load_models()

    predictions = predict.predict_next_14_days()
    for ptype, pdf in predictions.items():
        logger.info(f"  {ptype}: {len(pdf)} predictions generated")

    # ------------------------------------------------------------------
    # STEP 3 -- DECIDE AGENT
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 70)
    logger.info("  STEP 3: DECIDE AGENT -- Blocking Strategy")
    logger.info("-" * 70)

    decide = DecideAgent(predict, sense)
    blocking_decisions = decide.select_properties_to_block(week_start)
    logger.info(f"  {len(blocking_decisions)} properties selected for blocking")

    if blocking_decisions:
        top3 = blocking_decisions[:3]
        for d in top3:
            logger.info(
                f"    -> {d['property_id']} ({d['city']}): "
                f"score={d['total_score']:.3f}, rooms={d['rooms_to_block']}, "
                f"uplift=INR {d['expected_revenue_uplift']:,.0f}"
            )

    # ------------------------------------------------------------------
    # STEP 4 -- RESERVE AGENT
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 70)
    logger.info("  STEP 4: RESERVE AGENT -- Execute Blocking")
    logger.info("-" * 70)

    reserve = ReserveAgent(engine, decide)
    results = reserve.confirm_reservations()
    summary = reserve.get_execution_summary()
    logger.info(f"  Executed: {summary['total_blocks_executed']} blocks, "
                f"{summary['total_rooms_blocked']} rooms, "
                f"INR {summary['total_expected_uplift_inr']:,.0f} expected uplift")

    # ------------------------------------------------------------------
    # STEP 5 -- MONITOR AGENT
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 70)
    logger.info("  STEP 5: MONITOR AGENT -- Track Bookings")
    logger.info("-" * 70)

    monitor = MonitorAgent(engine)
    candidates = monitor.find_cheaper_rates_for_all(reference_date=week_start)
    logger.info(f"  Found {len(candidates)} rebooking candidates")

    # ------------------------------------------------------------------
    # STEP 6 -- OPTIMIZE AGENT
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 70)
    logger.info("  STEP 6: OPTIMIZE AGENT -- Smart Rebooking")
    logger.info("-" * 70)

    optimize = OptimizeAgent(monitor, engine)
    rebook_results = optimize.batch_process_rebookings(eval_week=week_start)
    opt_summary = optimize.get_optimization_summary()
    logger.info(f"  Rebooked: {opt_summary['total_rebooked']}, "
                f"Skipped: {opt_summary['total_skipped']}, "
                f"Margin recovered: INR {opt_summary['net_margin_recovered_inr']:,.0f}")

    # ------------------------------------------------------------------
    # STEP 7 -- REPORT AGENT
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 70)
    logger.info("  STEP 7: REPORT AGENT -- Weekly Executive Report")
    logger.info("-" * 70)

    report = ReportAgent(engine)
    report_path = report.create_executive_report(week_start, output_dir="reports")
    logger.info(f"  Report saved: {report_path}")

    # ------------------------------------------------------------------
    # Pipeline Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - pipeline_start
    logger.info("\n" + "=" * 70)
    logger.info("   PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total time: {elapsed:.1f}s")
    logger.info(f"  Properties blocked: {summary['total_blocks_executed']}")
    logger.info(f"  Rooms blocked: {summary['total_rooms_blocked']}")
    logger.info(f"  Expected uplift: INR {summary['total_expected_uplift_inr']:,.0f}")
    logger.info(f"  Rebookings: {opt_summary['total_rebooked']}")
    logger.info(f"  Margin recovered: INR {opt_summary['net_margin_recovered_inr']:,.0f}")
    logger.info(f"  Report: {report_path}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Agentic AI -- Demand Blocking & Smart Rebooking Pipeline"
    )
    parser.add_argument(
        "--week-start", type=str, default="2026-02-16",
        help="Start date of the target week (YYYY-MM-DD). Default: 2026-02-16",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Force retrain ML models even if saved models exist.",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Skip execution and only generate the weekly report.",
    )
    args = parser.parse_args()

    run_weekly_pipeline(
        week_start=args.week_start,
        force_train=args.train,
        report_only=args.report_only,
    )


if __name__ == "__main__":
    main()
