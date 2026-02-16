"""
Feature engineering utilities for ML model training.
Transforms raw database tables into ML-ready feature matrices.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_city_tier(tier: str) -> int:
    """Map city tier string to integer."""
    mapping = {"Tier1": 3, "Tier2": 2, "Tier3": 1}
    return mapping.get(tier, 0)


def encode_room_type(room_type: str) -> int:
    mapping = {"Standard": 1, "Deluxe": 2, "Suite": 3}
    return mapping.get(room_type, 0)


def encode_meal_plan(meal_plan: str) -> int:
    mapping = {"RO": 1, "BB": 2, "HB": 3, "FB": 4}
    return mapping.get(meal_plan, 0)


def encode_cancellation_type(cancel_type: str) -> int:
    mapping = {"NonRefundable": 1, "PartialRefund": 2, "Refundable": 3}
    return mapping.get(cancel_type, 0)


# ---------------------------------------------------------------------------
# Rolling / lag feature generators
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame, column: str, group_cols: list,
                     lags: list = None) -> pd.DataFrame:
    """Add lag features for a column grouped by group_cols."""
    if lags is None:
        lags = [7, 14, 30]
    df = df.sort_values(group_cols + ["date"]).copy()
    for lag in lags:
        col_name = f"{column}_lag_{lag}"
        df[col_name] = df.groupby(group_cols)[column].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, column: str, group_cols: list,
                         windows: list = None) -> pd.DataFrame:
    """Add rolling mean features for a column grouped by group_cols."""
    if windows is None:
        windows = [7, 14, 30]
    df = df.sort_values(group_cols + ["date"]).copy()
    for w in windows:
        col_name = f"{column}_rolling_{w}"
        df[col_name] = (
            df.groupby(group_cols)[column]
            .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
        )
    return df


def add_rolling_std(df: pd.DataFrame, column: str, group_cols: list,
                    windows: list = None) -> pd.DataFrame:
    """Add rolling standard deviation (volatility) features."""
    if windows is None:
        windows = [7, 14]
    df = df.sort_values(group_cols + ["date"]).copy()
    for w in windows:
        col_name = f"{column}_volatility_{w}d"
        df[col_name] = (
            df.groupby(group_cols)[column]
            .transform(lambda x: x.rolling(window=w, min_periods=2).std())
        )
    return df


# ---------------------------------------------------------------------------
# Calendar / temporal features
# ---------------------------------------------------------------------------

def add_temporal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add day-of-week, month, is_weekend features."""
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["day_of_month"] = dt.dt.day
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    return df


# ---------------------------------------------------------------------------
# Event proximity features
# ---------------------------------------------------------------------------

def add_event_features(df: pd.DataFrame, events_df: pd.DataFrame,
                       date_col: str = "date", city_col: str = "city") -> pd.DataFrame:
    """
    Add event-related features: days_to_event, event_demand_intensity.
    For each (city, date) find the nearest upcoming event and its intensity.
    """
    df = df.copy()
    events = events_df.copy()
    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events["end_date"])

    days_to_event = []
    event_intensities = []

    for _, row in df.iterrows():
        city = row[city_col]
        current_date = pd.to_datetime(row[date_col])
        city_events = events[events["city"] == city]

        if city_events.empty:
            days_to_event.append(999)
            event_intensities.append(0)
            continue

        # Check if currently during an event
        during = city_events[
            (city_events["start_date"] <= current_date) &
            (city_events["end_date"] >= current_date)
        ]
        if not during.empty:
            days_to_event.append(0)
            event_intensities.append(during["demand_intensity"].max())
            continue

        # Find nearest future event
        future = city_events[city_events["start_date"] > current_date]
        if future.empty:
            days_to_event.append(999)
            event_intensities.append(0)
        else:
            nearest = future.loc[(future["start_date"] - current_date).dt.days.idxmin()]
            days_to_event.append((nearest["start_date"] - current_date).days)
            event_intensities.append(nearest["demand_intensity"])

    df["days_to_event"] = days_to_event
    df["event_demand_intensity"] = event_intensities
    return df


def add_event_features_vectorized(df: pd.DataFrame, events_df: pd.DataFrame,
                                  date_col: str = "date",
                                  city_col: str = "city") -> pd.DataFrame:
    """
    Vectorized (fast) version of event feature engineering.
    Assigns days_to_event and event_demand_intensity for each row.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    events = events_df.copy()
    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events["end_date"])

    df["days_to_event"] = 999
    df["event_demand_intensity"] = 0

    for city in df[city_col].unique():
        city_mask = df[city_col] == city
        city_events = events[events["city"] == city]
        if city_events.empty:
            continue

        city_dates = df.loc[city_mask, date_col]
        for _, ev in city_events.iterrows():
            # During event
            during_mask = city_mask & (df[date_col] >= ev["start_date"]) & (df[date_col] <= ev["end_date"])
            df.loc[during_mask, "days_to_event"] = 0
            df.loc[during_mask, "event_demand_intensity"] = np.maximum(
                df.loc[during_mask, "event_demand_intensity"], ev["demand_intensity"]
            )

            # Before event
            before_mask = city_mask & (df[date_col] < ev["start_date"])
            if before_mask.any():
                days_diff = (ev["start_date"] - df.loc[before_mask, date_col]).dt.days
                closer = days_diff < df.loc[before_mask, "days_to_event"]
                update_idx = closer[closer].index
                df.loc[update_idx, "days_to_event"] = days_diff[closer].values
                df.loc[update_idx, "event_demand_intensity"] = ev["demand_intensity"]

    return df


# ---------------------------------------------------------------------------
# Occupancy & booking features for Property_Daily
# ---------------------------------------------------------------------------

def add_occupancy_features(property_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Derive occupancy rate, conversion rate, booking velocity and their
    rolling aggregates from Property_Daily.
    """
    df = property_daily.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Occupancy rate
    df["occupancy_rate"] = np.where(
        df["base_inventory_rooms"] > 0,
        df["rooms_sold"] / df["base_inventory_rooms"],
        0.0,
    )

    # Conversion rate
    df["conversion_rate"] = np.where(
        df["booking_requests"] > 0,
        df["rooms_sold"] / df["booking_requests"],
        0.0,
    )

    # Booking velocity (requests per day — already daily, so it's just the value)
    df["booking_velocity"] = df["booking_requests"]

    # Rolling features
    group = ["property_id"]
    df = add_rolling_features(df, "occupancy_rate", group, [7, 14])
    df = add_rolling_features(df, "booking_velocity", group, [7])
    df = add_rolling_features(df, "conversion_rate", group, [7])

    # Lag features
    df = add_lag_features(df, "occupancy_rate", group, [7, 14])
    df = add_lag_features(df, "rooms_sold", group, [7, 14, 30])

    # Sell-out frequency (rolling count of sold_out_flag in last 30 days)
    df = df.sort_values(group + ["date"])
    df["soldout_freq_30d"] = (
        df.groupby("property_id")["sold_out_flag"]
        .transform(lambda x: x.rolling(window=30, min_periods=1).sum())
    )

    return df


# ---------------------------------------------------------------------------
# Supplier features
# ---------------------------------------------------------------------------

def merge_supplier_features(df: pd.DataFrame,
                            supplier_df: pd.DataFrame,
                            supplier_col: str = "supplier_id") -> pd.DataFrame:
    """Merge supplier reliability metrics into a DataFrame."""
    sup = supplier_df[["supplier_id", "booking_failure_rate",
                        "supplier_cancellation_rate",
                        "avg_confirmation_time_mins",
                        "dispute_rate", "preferred_supplier_flag"]].copy()
    sup = sup.rename(columns={
        "booking_failure_rate": "supplier_failure_rate",
        "supplier_cancellation_rate": "supplier_cancel_rate",
        "preferred_supplier_flag": "supplier_preferred",
    })
    return df.merge(sup, left_on=supplier_col, right_on="supplier_id", how="left")


# ---------------------------------------------------------------------------
# Price features from Rate_Snapshots
# ---------------------------------------------------------------------------

def add_price_features(rate_snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag, rolling, and volatility features for net_rate_inr.
    """
    df = rate_snapshots.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df = df.rename(columns={"snapshot_date": "date"})

    group = ["property_id", "supplier_id", "standard_room_type", "meal_plan"]
    df = df.sort_values(group + ["date"])

    df = add_lag_features(df, "net_rate_inr", group, [7, 14])
    df = add_rolling_features(df, "net_rate_inr", group, [7, 14])
    df = add_rolling_std(df, "net_rate_inr", group, [7])

    # Rename volatility column for config compatibility
    if "net_rate_inr_volatility_7d" in df.columns:
        df = df.rename(columns={"net_rate_inr_volatility_7d": "price_volatility_7d"})

    return df


# ---------------------------------------------------------------------------
# Lead-time features from Confirmed_Bookings
# ---------------------------------------------------------------------------

def add_lead_time_features(bookings: pd.DataFrame) -> pd.DataFrame:
    """Compute lead time (days between booking and check-in)."""
    df = bookings.copy()
    df["booked_date"] = pd.to_datetime(df["booked_date"])
    df["stay_checkin_date"] = pd.to_datetime(df["stay_checkin_date"])
    df["lead_time_days"] = (df["stay_checkin_date"] - df["booked_date"]).dt.days
    return df


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

def build_demand_features(city_demand: pd.DataFrame,
                          events_df: pd.DataFrame,
                          property_master: pd.DataFrame,
                          weekly_demand: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix for the Demand Spike prediction model.
    Target: city_demand_multiplier (future value).
    """
    df = city_demand.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Temporal features
    df = add_temporal_features(df, "date")

    # Lag and rolling features on demand multiplier
    df = add_lag_features(df, "city_demand_multiplier", ["city"], [7, 14, 30])
    df = add_rolling_features(df, "city_demand_multiplier", ["city"], [7, 14, 30])

    # Event features
    df = add_event_features_vectorized(df, events_df, "date", "city")

    # City tier encoding
    city_tier_map = property_master.drop_duplicates("city").set_index("city")["city_tier"]
    df["city_tier_encoded"] = df["city"].map(city_tier_map).apply(encode_city_tier)

    # Merge weekly demand for total_requests_lag_7
    if weekly_demand is not None and not weekly_demand.empty:
        wd = weekly_demand.copy()
        wd["week_start"] = pd.to_datetime(wd["week_start"])
        wd = wd.rename(columns={"total_requests": "total_requests_weekly"})
        wd["date_key"] = wd["week_start"]
        df["date_key"] = df["date"].dt.to_period("W").dt.start_time
        df = df.merge(wd[["date_key", "city", "total_requests_weekly"]],
                      on=["date_key", "city"], how="left")
        df["total_requests_lag_7"] = df.groupby("city")["total_requests_weekly"].shift(1)
        df.drop(columns=["date_key", "total_requests_weekly"], inplace=True, errors="ignore")
    else:
        df["total_requests_lag_7"] = 0

    return df


def build_sellout_features(property_daily: pd.DataFrame,
                           events_df: pd.DataFrame,
                           property_master: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix for the Sell-out Probability model.
    Target: sold_out_flag (binary).
    """
    df = add_occupancy_features(property_daily)
    df = add_temporal_features(df, "date")
    df = add_event_features_vectorized(df, events_df, "date", "city")

    # City tier
    city_tier_map = property_master.drop_duplicates("city").set_index("city")["city_tier"]
    df["city_tier_encoded"] = df["city"].map(city_tier_map).apply(encode_city_tier)

    return df


def build_price_features(rate_snapshots: pd.DataFrame,
                         city_demand: pd.DataFrame,
                         events_df: pd.DataFrame,
                         property_master: pd.DataFrame,
                         supplier_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix for the Price Movement model.
    Target: net_rate_inr (future value).
    """
    df = add_price_features(rate_snapshots)
    df = add_temporal_features(df, "date")

    # Merge demand signals
    cd = city_demand[["city", "date", "city_demand_multiplier",
                      "seasonality", "weekend_multiplier", "event_multiplier"]].copy()
    cd["date"] = pd.to_datetime(cd["date"])
    df = df.merge(cd, on=["city", "date"], how="left")

    # Event features
    df = add_event_features_vectorized(df, events_df, "date", "city")

    # Property features
    pm = property_master[["property_id", "star_rating", "popularity_index",
                          "base_adr_inr", "city_tier"]].copy()
    pm["city_tier_encoded"] = pm["city_tier"].apply(encode_city_tier)
    df = df.merge(pm.drop(columns=["city_tier"]), on="property_id", how="left")

    # Supplier features
    df = merge_supplier_features(df, supplier_df)

    # Encode categoricals
    df["room_type_encoded"] = df["standard_room_type"].apply(encode_room_type)
    df["meal_plan_encoded"] = df["meal_plan"].apply(encode_meal_plan)

    # Rename lag columns for config compatibility
    rename_map = {
        "net_rate_inr_lag_7": "net_rate_lag_7",
        "net_rate_inr_lag_14": "net_rate_lag_14",
        "net_rate_inr_rolling_7": "net_rate_rolling_7",
        "net_rate_inr_rolling_14": "net_rate_rolling_14",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df
