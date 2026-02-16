"""
SENSE AGENT - Data Ingestion & Feature Engineering
===================================================
Responsibilities:
  1. Connect to MS SQL Server and ingest all 12 tables
  2. Validate data quality and handle missing values
  3. Engineer ML-ready features for downstream agents
  4. Provide current-state snapshots for decision making
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

from utils.db_utils import get_sqlalchemy_engine, read_all_tables, read_table, test_connection
from utils.feature_engineering import (
    build_demand_features,
    build_sellout_features,
    build_price_features,
    add_lead_time_features,
    add_temporal_features,
    encode_city_tier,
)

logger = logging.getLogger(__name__)


class SenseAgent:
    """
    Agent responsible for sensing the environment: data ingestion,
    quality checks, and feature engineering.
    """

    def __init__(self, engine=None):
        """
        Parameters
        ----------
        engine : sqlalchemy.Engine, optional
            Pre-created engine; one is created if not provided.
        """
        self.engine = engine or get_sqlalchemy_engine()
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.features: Dict[str, pd.DataFrame] = {}
        self._ingested = False

    # ------------------------------------------------------------------
    # 1. Data Ingestion
    # ------------------------------------------------------------------

    def ingest_data(self) -> Dict[str, pd.DataFrame]:
        """Load all 12 tables from the database."""
        logger.info("=" * 60)
        logger.info("SENSE AGENT: Starting data ingestion")
        logger.info("=" * 60)

        if not test_connection(self.engine):
            raise ConnectionError("Cannot connect to database.")

        self.raw_data = read_all_tables(self.engine)
        self._ingested = True

        # Log summary
        total_rows = sum(len(df) for df in self.raw_data.values())
        logger.info(f"Ingested {len(self.raw_data)} tables, {total_rows:,} total rows.")
        for key, df in self.raw_data.items():
            logger.info(f"  {key}: {len(df)} rows x {len(df.columns)} cols")

        # Convert date columns
        self._convert_dates()

        return self.raw_data

    def _convert_dates(self):
        """Ensure all date columns are proper datetime types."""
        date_cols_map = {
            "city_demand_signals": ["date"],
            "property_daily": ["date"],
            "events_calendar": ["start_date", "end_date"],
            "rate_snapshots": ["snapshot_date"],
            "confirmed_bookings": ["booked_date", "stay_checkin_date",
                                   "stay_checkout_date", "cancel_deadline_date"],
            "demand_block_actions": ["week_start", "block_start_date", "block_end_date"],
            "rebooking_evaluations": ["evaluation_date", "eval_week"],
            "weekly_demand_bycity": ["week_start"],
            "weekly_kpi_summary": ["week_start"],
        }
        for table_key, cols in date_cols_map.items():
            if table_key in self.raw_data:
                for col in cols:
                    if col in self.raw_data[table_key].columns:
                        self.raw_data[table_key][col] = pd.to_datetime(
                            self.raw_data[table_key][col], errors="coerce"
                        )

    # ------------------------------------------------------------------
    # 2. Data Quality
    # ------------------------------------------------------------------

    def validate_data(self) -> Dict[str, dict]:
        """
        Run quality checks on ingested data and return a report.
        """
        self._ensure_ingested()
        report = {}
        for key, df in self.raw_data.items():
            missing = df.isnull().sum()
            missing_pct = (missing / len(df) * 100).round(2)
            cols_with_missing = missing[missing > 0]
            report[key] = {
                "rows": len(df),
                "columns": len(df.columns),
                "missing_columns": cols_with_missing.to_dict(),
                "missing_pct": missing_pct[missing_pct > 0].to_dict(),
                "duplicates": df.duplicated().sum(),
            }
            if cols_with_missing.any():
                logger.warning(
                    f"Table {key} has missing values: "
                    f"{cols_with_missing.to_dict()}"
                )
        return report

    def handle_missing_data(self):
        """Impute or drop missing values based on column semantics."""
        self._ensure_ingested()
        logger.info("Handling missing data ...")

        # Numeric columns: fill with median per group or global median
        numeric_tables = ["property_daily", "city_demand_signals", "rate_snapshots"]
        for tkey in numeric_tables:
            if tkey in self.raw_data:
                df = self.raw_data[tkey]
                num_cols = df.select_dtypes(include=[np.number]).columns
                for col in num_cols:
                    if df[col].isnull().any():
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                        logger.info(f"  Filled {tkey}.{col} nulls with median={median_val:.2f}")

        # Categorical columns: fill with mode
        for tkey, df in self.raw_data.items():
            cat_cols = df.select_dtypes(include=["object"]).columns
            for col in cat_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col].fillna(mode_val[0], inplace=True)
                        logger.info(f"  Filled {tkey}.{col} nulls with mode={mode_val[0]}")

    # ------------------------------------------------------------------
    # 3. Feature Engineering
    # ------------------------------------------------------------------

    def engineer_features(self) -> Dict[str, pd.DataFrame]:
        """
        Create ML-ready feature matrices for all three models.

        Returns
        -------
        dict with keys: 'demand_features', 'sellout_features', 'price_features'
        """
        self._ensure_ingested()
        logger.info("=" * 60)
        logger.info("SENSE AGENT: Engineering features")
        logger.info("=" * 60)

        # Demand features
        logger.info("Building demand spike features ...")
        self.features["demand_features"] = build_demand_features(
            city_demand=self.raw_data["city_demand_signals"],
            events_df=self.raw_data["events_calendar"],
            property_master=self.raw_data["property_master"],
            weekly_demand=self.raw_data.get("weekly_demand_bycity"),
        )
        logger.info(f"  -> {self.features['demand_features'].shape}")

        # Sell-out features
        logger.info("Building sell-out probability features ...")
        self.features["sellout_features"] = build_sellout_features(
            property_daily=self.raw_data["property_daily"],
            events_df=self.raw_data["events_calendar"],
            property_master=self.raw_data["property_master"],
        )
        logger.info(f"  -> {self.features['sellout_features'].shape}")

        # Price features
        logger.info("Building price movement features ...")
        self.features["price_features"] = build_price_features(
            rate_snapshots=self.raw_data["rate_snapshots"],
            city_demand=self.raw_data["city_demand_signals"],
            events_df=self.raw_data["events_calendar"],
            property_master=self.raw_data["property_master"],
            supplier_df=self.raw_data["supplier_reliability"],
        )
        logger.info(f"  -> {self.features['price_features'].shape}")

        return self.features

    # ------------------------------------------------------------------
    # 4. Getters for downstream agents
    # ------------------------------------------------------------------

    def get_training_data(self) -> Dict[str, pd.DataFrame]:
        """Return engineered features for model training."""
        if not self.features:
            self.engineer_features()
        return self.features

    def get_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Return raw ingested tables."""
        self._ensure_ingested()
        return self.raw_data

    def get_current_state(self, reference_date: Optional[str] = None) -> dict:
        """
        Get the latest snapshot of key metrics for decision making.

        Parameters
        ----------
        reference_date : str, optional
            Date to use as 'today'. Defaults to max date in Property_Daily.

        Returns
        -------
        dict with current metrics per city and property.
        """
        self._ensure_ingested()

        pd_df = self.raw_data["property_daily"]
        if reference_date is None:
            reference_date = pd_df["date"].max()
        else:
            reference_date = pd.to_datetime(reference_date)

        # Latest property-level metrics
        latest = pd_df[pd_df["date"] == reference_date].copy()
        latest["occupancy_rate"] = np.where(
            latest["base_inventory_rooms"] > 0,
            latest["rooms_sold"] / latest["base_inventory_rooms"],
            0.0,
        )

        # Latest city-level demand signals
        cd = self.raw_data["city_demand_signals"]
        latest_cd = cd[cd["date"] == cd["date"].max()]

        return {
            "reference_date": str(reference_date.date()) if hasattr(reference_date, "date") else str(reference_date),
            "property_snapshot": latest,
            "city_demand_snapshot": latest_cd,
            "total_properties": len(latest),
            "avg_occupancy": round(latest["occupancy_rate"].mean(), 3),
            "soldout_count": int(latest["sold_out_flag"].sum()),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_ingested(self):
        if not self._ingested:
            raise RuntimeError("Data not ingested yet. Call ingest_data() first.")
