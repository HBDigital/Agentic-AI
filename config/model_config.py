"""
ML model hyperparameters and training configuration.
All values are data-driven defaults; tune after initial training.
"""

# ---------------------------------------------------------------------------
# General training settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
PREDICTION_HORIZON_DAYS = 14  # predict next 14 days

# ---------------------------------------------------------------------------
# Demand Spike Model (XGBoost Regressor)
# Target: city_demand_multiplier
# ---------------------------------------------------------------------------
DEMAND_MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

DEMAND_MODEL_FEATURES = [
    "seasonality",
    "weekend_multiplier",
    "event_multiplier",
    "city_demand_multiplier_lag_7",
    "city_demand_multiplier_lag_14",
    "city_demand_multiplier_lag_30",
    "city_demand_multiplier_rolling_7",
    "city_demand_multiplier_rolling_14",
    "city_demand_multiplier_rolling_30",
    "day_of_week",
    "month",
    "is_weekend",
    "city_tier_encoded",
    "event_demand_intensity",
    "total_requests_lag_7",
]

# ---------------------------------------------------------------------------
# Sell-out Probability Model (XGBoost Classifier)
# Target: sold_out_flag (binary)
# ---------------------------------------------------------------------------
SELLOUT_MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": 3.0,  # adjusted for class imbalance
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "auc",
}

SELLOUT_MODEL_FEATURES = [
    "star_rating",
    "base_inventory_rooms",
    "popularity_index",
    "occupancy_rate",
    "occupancy_rate_rolling_7",
    "occupancy_rate_rolling_14",
    "booking_velocity",
    "booking_velocity_rolling_7",
    "conversion_rate",
    "city_demand_multiplier",
    "seasonality",
    "weekend_multiplier",
    "event_multiplier",
    "day_of_week",
    "month",
    "is_weekend",
    "city_tier_encoded",
    "soldout_freq_30d",
    "days_to_event",
    "base_adr_inr",
]

# ---------------------------------------------------------------------------
# Price Movement Model (XGBoost Regressor)
# Target: net_rate_inr
# ---------------------------------------------------------------------------
PRICE_MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

PRICE_MODEL_FEATURES = [
    "base_adr_inr",
    "star_rating",
    "popularity_index",
    "city_demand_multiplier",
    "seasonality",
    "weekend_multiplier",
    "event_multiplier",
    "room_type_encoded",
    "meal_plan_encoded",
    "supplier_failure_rate",
    "supplier_preferred",
    "net_rate_lag_7",
    "net_rate_lag_14",
    "net_rate_rolling_7",
    "net_rate_rolling_14",
    "price_volatility_7d",
    "day_of_week",
    "month",
    "is_weekend",
    "city_tier_encoded",
]

# ---------------------------------------------------------------------------
# Model file paths (relative to project root)
# ---------------------------------------------------------------------------
MODEL_DIR = "models"
DEMAND_MODEL_PATH = f"{MODEL_DIR}/demand_model.pkl"
SELLOUT_MODEL_PATH = f"{MODEL_DIR}/sellout_model.pkl"
PRICE_MODEL_PATH = f"{MODEL_DIR}/price_model.pkl"
FEATURE_SCALER_PATH = f"{MODEL_DIR}/feature_scaler.pkl"
LABEL_ENCODERS_PATH = f"{MODEL_DIR}/label_encoders.pkl"
