"""
PREDICT AGENT - ML Model Training & Prediction
================================================
Responsibilities:
  1. Train 3 ML models: Demand Spike, Sell-out Probability, Price Movement
  2. Validate models with cross-validation and holdout metrics
  3. Generate predictions for the next 14 days
  4. Save/load trained models for reuse
"""
import logging
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config.model_config import (
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, PREDICTION_HORIZON_DAYS,
    DEMAND_MODEL_PARAMS, DEMAND_MODEL_FEATURES,
    SELLOUT_MODEL_PARAMS, SELLOUT_MODEL_FEATURES,
    PRICE_MODEL_PARAMS, PRICE_MODEL_FEATURES,
    DEMAND_MODEL_PATH, SELLOUT_MODEL_PATH, PRICE_MODEL_PATH,
    FEATURE_SCALER_PATH, LABEL_ENCODERS_PATH, MODEL_DIR,
)
from utils.evaluation import (
    evaluate_regression, evaluate_classification,
    cross_validate_model, get_feature_importance, backtest_predictions,
)

logger = logging.getLogger(__name__)


class PredictAgent:
    """
    Agent responsible for training ML models and generating predictions
    for demand spikes, sell-out probabilities, and price movements.
    """

    def __init__(self, sense_agent, project_root: str = "."):
        """
        Parameters
        ----------
        sense_agent : SenseAgent
            Provides engineered features.
        project_root : str
            Root directory for saving/loading model artifacts.
        """
        self.sense = sense_agent
        self.project_root = project_root
        self.features = sense_agent.get_training_data()
        self.raw_data = sense_agent.get_raw_data()

        # Models
        self.demand_model = None
        self.sellout_model = None
        self.price_model = None
        self.scaler = StandardScaler()

        # Metrics storage
        self.metrics: Dict[str, dict] = {}
        self.feature_importances: Dict[str, pd.DataFrame] = {}

        # Ensure model directory exists
        model_dir = os.path.join(project_root, MODEL_DIR)
        os.makedirs(model_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Model 1: Demand Spike Prediction
    # ------------------------------------------------------------------

    def train_demand_model(self) -> dict:
        """
        Train XGBoost regressor to predict city_demand_multiplier.
        Returns evaluation metrics.
        """
        logger.info("=" * 60)
        logger.info("PREDICT AGENT: Training Demand Spike Model")
        logger.info("=" * 60)

        df = self.features["demand_features"].copy()

        # Prepare target: shift demand multiplier forward as prediction target
        df = df.sort_values(["city", "date"])
        df["target"] = df.groupby("city")["city_demand_multiplier"].shift(
            -PREDICTION_HORIZON_DAYS
        )
        df = df.dropna(subset=["target"])

        # Select features that exist in the dataframe
        available_features = [f for f in DEMAND_MODEL_FEATURES if f in df.columns]
        missing = set(DEMAND_MODEL_FEATURES) - set(available_features)
        if missing:
            logger.warning(f"Missing demand features: {missing}")

        X = df[available_features].fillna(0)
        y = df["target"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Train model
        self.demand_model = XGBRegressor(**DEMAND_MODEL_PARAMS)
        self.demand_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = self.demand_model.predict(X_test)
        self.metrics["demand"] = evaluate_regression(y_test, y_pred, "Demand Spike Model")

        # Cross-validation
        cv_results = cross_validate_model(
            self.demand_model, X, y, cv=CV_FOLDS,
            scoring="neg_mean_absolute_error",
            model_name="Demand Spike Model",
        )
        self.metrics["demand"]["cv"] = cv_results

        # Feature importance
        self.feature_importances["demand"] = get_feature_importance(
            self.demand_model, available_features
        )
        self._demand_features = available_features

        logger.info(f"Demand model trained. MAE={self.metrics['demand']['MAE']}")
        return self.metrics["demand"]

    # ------------------------------------------------------------------
    # Model 2: Sell-out Probability Prediction
    # ------------------------------------------------------------------

    def train_sellout_model(self) -> dict:
        """
        Train XGBoost classifier to predict sold_out_flag.
        Returns evaluation metrics.
        """
        logger.info("=" * 60)
        logger.info("PREDICT AGENT: Training Sell-out Probability Model")
        logger.info("=" * 60)

        df = self.features["sellout_features"].copy()

        # Target: sell-out in next N days (forward-looking)
        df = df.sort_values(["property_id", "date"])
        df["target"] = df.groupby("property_id")["sold_out_flag"].transform(
            lambda x: x.rolling(window=PREDICTION_HORIZON_DAYS, min_periods=1).max().shift(-PREDICTION_HORIZON_DAYS)
        )
        df = df.dropna(subset=["target"])
        df["target"] = df["target"].astype(int)

        # Select available features
        available_features = [f for f in SELLOUT_MODEL_FEATURES if f in df.columns]
        missing = set(SELLOUT_MODEL_FEATURES) - set(available_features)
        if missing:
            logger.warning(f"Missing sellout features: {missing}")

        X = df[available_features].fillna(0)
        y = df["target"]

        # Handle class imbalance - calculate scale_pos_weight dynamically
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        if pos_count > 0:
            dynamic_scale = neg_count / pos_count
        else:
            dynamic_scale = 1.0

        params = SELLOUT_MODEL_PARAMS.copy()
        params["scale_pos_weight"] = min(dynamic_scale, 10.0)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        # Train model
        self.sellout_model = XGBClassifier(**params)
        self.sellout_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = self.sellout_model.predict(X_test)
        y_prob = self.sellout_model.predict_proba(X_test)[:, 1]
        self.metrics["sellout"] = evaluate_classification(
            y_test, y_pred, y_prob, "Sell-out Probability Model"
        )

        # Cross-validation
        cv_results = cross_validate_model(
            self.sellout_model, X, y, cv=CV_FOLDS,
            scoring="roc_auc",
            model_name="Sell-out Probability Model",
        )
        self.metrics["sellout"]["cv"] = cv_results

        # Feature importance
        self.feature_importances["sellout"] = get_feature_importance(
            self.sellout_model, available_features
        )
        self._sellout_features = available_features

        logger.info(f"Sellout model trained. AUC={self.metrics['sellout'].get('AUC_ROC', 'N/A')}")
        return self.metrics["sellout"]

    # ------------------------------------------------------------------
    # Model 3: Price Movement Prediction
    # ------------------------------------------------------------------

    def train_price_model(self) -> dict:
        """
        Train XGBoost regressor to predict future net_rate_inr.
        Returns evaluation metrics.
        """
        logger.info("=" * 60)
        logger.info("PREDICT AGENT: Training Price Movement Model")
        logger.info("=" * 60)

        df = self.features["price_features"].copy()

        # Target: future net rate (shifted forward)
        group_cols = ["property_id", "supplier_id", "standard_room_type", "meal_plan"]
        existing_groups = [c for c in group_cols if c in df.columns]
        df = df.sort_values(existing_groups + ["date"])

        if existing_groups:
            df["target"] = df.groupby(existing_groups)["net_rate_inr"].shift(
                -PREDICTION_HORIZON_DAYS
            )
        else:
            df["target"] = df["net_rate_inr"].shift(-PREDICTION_HORIZON_DAYS)

        df = df.dropna(subset=["target"])

        # Select available features
        available_features = [f for f in PRICE_MODEL_FEATURES if f in df.columns]
        missing = set(PRICE_MODEL_FEATURES) - set(available_features)
        if missing:
            logger.warning(f"Missing price features: {missing}")

        X = df[available_features].fillna(0)
        y = df["target"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Train model
        self.price_model = XGBRegressor(**PRICE_MODEL_PARAMS)
        self.price_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = self.price_model.predict(X_test)
        self.metrics["price"] = evaluate_regression(y_test, y_pred, "Price Movement Model")

        # Cross-validation
        cv_results = cross_validate_model(
            self.price_model, X, y, cv=CV_FOLDS,
            scoring="neg_mean_absolute_error",
            model_name="Price Movement Model",
        )
        self.metrics["price"]["cv"] = cv_results

        # Feature importance
        self.feature_importances["price"] = get_feature_importance(
            self.price_model, available_features
        )
        self._price_features = available_features

        logger.info(f"Price model trained. MAPE={self.metrics['price']['MAPE_%']}%")
        return self.metrics["price"]

    # ------------------------------------------------------------------
    # Train All Models
    # ------------------------------------------------------------------

    def train_all_models(self) -> Dict[str, dict]:
        """Train all three models and return combined metrics."""
        self.train_demand_model()
        self.train_sellout_model()
        self.train_price_model()
        return self.metrics

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def predict_demand(self, city: str = None,
                       start_date: str = None) -> pd.DataFrame:
        """
        Predict demand multiplier for next 14 days.

        Returns DataFrame with columns:
            city, date, predicted_demand_index, confidence
        """
        if self.demand_model is None:
            raise RuntimeError("Demand model not trained. Call train_demand_model() first.")

        df = self.features["demand_features"].copy()
        if city:
            df = df[df["city"] == city]

        # Use the latest available row per city as input for forward prediction
        df = df.sort_values("date")
        recent = df.groupby("city").tail(14)  # last 14 days per city

        features = [f for f in self._demand_features if f in recent.columns]
        X = recent[features].fillna(0)

        predictions = recent[["city", "date"]].copy()
        pred_values = self.demand_model.predict(X)
        predictions["predicted_demand_index"] = np.round(pred_values, 3)

        # Confidence based on prediction variance from training
        train_preds = self.demand_model.predict(
            self.features["demand_features"][features].fillna(0).tail(1000)
        )
        pred_std = np.std(train_preds)
        predictions["confidence"] = np.clip(
            1.0 - np.abs(pred_values - np.mean(train_preds)) / (3 * pred_std + 1e-6),
            0.5, 0.99
        ).round(3)

        return predictions.reset_index(drop=True)

    def predict_sellout(self, property_id: str = None) -> pd.DataFrame:
        """
        Predict sell-out probability for next 14 days.

        Returns DataFrame with columns:
            property_id, date, sellout_probability, expected_booking_surge
        """
        if self.sellout_model is None:
            raise RuntimeError("Sellout model not trained. Call train_sellout_model() first.")

        df = self.features["sellout_features"].copy()
        if property_id:
            df = df[df["property_id"] == property_id]

        # Use last 14 days per property for forward prediction
        df = df.sort_values("date")
        recent = df.groupby("property_id").tail(14)

        features = [f for f in self._sellout_features if f in recent.columns]
        X = recent[features].fillna(0)

        predictions = recent[["property_id", "city", "date"]].copy()
        prob = self.sellout_model.predict_proba(X)[:, 1]
        predictions["sellout_probability"] = np.round(prob, 3)

        # Expected booking surge from booking velocity
        if "booking_velocity" in recent.columns:
            predictions["expected_booking_surge"] = recent["booking_velocity"].values
        else:
            predictions["expected_booking_surge"] = 0

        # Estimate days until sellout based on occupancy trend
        if "occupancy_rate" in recent.columns and "occupancy_rate_rolling_7" in recent.columns:
            occ_rate = recent["occupancy_rate"].values
            occ_trend = recent["occupancy_rate_rolling_7"].values
            daily_increase = np.clip(occ_rate - occ_trend, 0.001, 0.5)
            remaining = np.clip(1.0 - occ_rate, 0, 1.0)
            predictions["days_until_sellout"] = np.where(
                prob > 0.5,
                np.clip(np.ceil(remaining / daily_increase), 1, 30).astype(int),
                99
            )
        else:
            predictions["days_until_sellout"] = 99

        return predictions.reset_index(drop=True)

    def predict_prices(self, property_id: str = None,
                       supplier_id: str = None) -> pd.DataFrame:
        """
        Predict future net_rate_inr.

        Returns DataFrame with columns:
            property_id, supplier_id, room_type, current_rate,
            predicted_rate_7d, price_trend, volatility
        """
        if self.price_model is None:
            raise RuntimeError("Price model not trained. Call train_price_model() first.")

        df = self.features["price_features"].copy()
        if property_id:
            df = df[df["property_id"] == property_id]
        if supplier_id:
            df = df[df["supplier_id"] == supplier_id]

        # Use last 14 days per property-supplier for forward prediction
        df = df.sort_values("date")
        group_cols = [c for c in ["property_id", "supplier_id"] if c in df.columns]
        recent = df.groupby(group_cols).tail(14) if group_cols else df.tail(500)

        features = [f for f in self._price_features if f in recent.columns]
        X = recent[features].fillna(0)

        predictions = recent[["property_id", "supplier_id",
                              "standard_room_type", "net_rate_inr"]].copy()
        predictions = predictions.rename(columns={
            "standard_room_type": "room_type",
            "net_rate_inr": "current_rate",
        })
        pred_rate = self.price_model.predict(X)
        predictions["predicted_rate_7d"] = np.round(pred_rate, 2)

        # Price trend
        predictions["price_trend"] = np.where(
            pred_rate > predictions["current_rate"] * 1.02, "increasing",
            np.where(pred_rate < predictions["current_rate"] * 0.98, "decreasing", "stable")
        )

        # Volatility
        if "price_volatility_7d" in recent.columns:
            predictions["volatility"] = recent["price_volatility_7d"].values.round(3)
        else:
            predictions["volatility"] = 0.0

        return predictions.reset_index(drop=True)

    def predict_next_14_days(self) -> Dict[str, pd.DataFrame]:
        """
        Run all three prediction models and return combined results.
        """
        logger.info("=" * 60)
        logger.info("PREDICT AGENT: Generating 14-day predictions")
        logger.info("=" * 60)

        results = {}

        try:
            results["demand"] = self.predict_demand()
            logger.info(f"  Demand predictions: {len(results['demand'])} rows")
        except Exception as e:
            logger.error(f"Demand prediction failed: {e}")
            results["demand"] = pd.DataFrame()

        try:
            results["sellout"] = self.predict_sellout()
            logger.info(f"  Sellout predictions: {len(results['sellout'])} rows")
        except Exception as e:
            logger.error(f"Sellout prediction failed: {e}")
            results["sellout"] = pd.DataFrame()

        try:
            results["prices"] = self.predict_prices()
            logger.info(f"  Price predictions: {len(results['prices'])} rows")
        except Exception as e:
            logger.error(f"Price prediction failed: {e}")
            results["prices"] = pd.DataFrame()

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_models(self):
        """Save trained models to disk."""
        root = self.project_root
        artifacts = {
            os.path.join(root, DEMAND_MODEL_PATH): self.demand_model,
            os.path.join(root, SELLOUT_MODEL_PATH): self.sellout_model,
            os.path.join(root, PRICE_MODEL_PATH): self.price_model,
        }
        for path, model in artifacts.items():
            if model is not None:
                with open(path, "wb") as f:
                    pickle.dump(model, f)
                logger.info(f"Saved model to {path}")

        # Save feature lists
        meta = {
            "demand_features": getattr(self, "_demand_features", []),
            "sellout_features": getattr(self, "_sellout_features", []),
            "price_features": getattr(self, "_price_features", []),
        }
        meta_path = os.path.join(root, LABEL_ENCODERS_PATH)
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        logger.info(f"Saved feature metadata to {meta_path}")

    def load_models(self):
        """Load previously trained models from disk."""
        root = self.project_root
        paths = {
            "demand": os.path.join(root, DEMAND_MODEL_PATH),
            "sellout": os.path.join(root, SELLOUT_MODEL_PATH),
            "price": os.path.join(root, PRICE_MODEL_PATH),
        }
        for key, path in paths.items():
            if os.path.exists(path):
                with open(path, "rb") as f:
                    model = pickle.load(f)
                setattr(self, f"{key}_model", model)
                logger.info(f"Loaded {key} model from {path}")

        # Load feature metadata
        meta_path = os.path.join(root, LABEL_ENCODERS_PATH)
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self._demand_features = meta.get("demand_features", [])
            self._sellout_features = meta.get("sellout_features", [])
            self._price_features = meta.get("price_features", [])
            logger.info("Loaded feature metadata.")
