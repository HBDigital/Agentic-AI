"""
Model evaluation utilities.
Provides metrics computation, cross-validation, and reporting helpers.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


def evaluate_regression(y_true, y_pred, model_name: str = "Model") -> dict:
    """
    Evaluate a regression model and return metrics dictionary.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    metrics = {
        "model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE_%": round(mape, 2),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Regression Metrics for {model_name}")
    logger.info(f"{'='*50}")
    for k, v in metrics.items():
        if k != "model":
            logger.info(f"  {k}: {v}")
    logger.info(f"{'='*50}\n")

    return metrics


def evaluate_classification(y_true, y_pred, y_prob=None,
                            model_name: str = "Model") -> dict:
    """
    Evaluate a binary classification model and return metrics dictionary.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4),
    }

    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        metrics["AUC_ROC"] = round(auc, 4)

    cm = confusion_matrix(y_true, y_pred)
    metrics["Confusion_Matrix"] = cm.tolist()

    logger.info(f"\n{'='*50}")
    logger.info(f"Classification Metrics for {model_name}")
    logger.info(f"{'='*50}")
    for k, v in metrics.items():
        if k not in ("model", "Confusion_Matrix"):
            logger.info(f"  {k}: {v}")
    logger.info(f"  Confusion Matrix:\n{cm}")
    logger.info(f"{'='*50}\n")

    return metrics


def cross_validate_model(model, X, y, cv: int = 5,
                         scoring: str = "neg_mean_absolute_error",
                         model_name: str = "Model") -> dict:
    """
    Run cross-validation and return summary statistics.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    if scoring.startswith("neg_"):
        scores = -scores
        metric_name = scoring.replace("neg_", "")
    else:
        metric_name = scoring

    results = {
        "model": model_name,
        "metric": metric_name,
        "cv_folds": cv,
        "mean": round(np.mean(scores), 4),
        "std": round(np.std(scores), 4),
        "min": round(np.min(scores), 4),
        "max": round(np.max(scores), 4),
        "scores": [round(s, 4) for s in scores],
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Cross-Validation for {model_name}")
    logger.info(f"  Metric: {metric_name}")
    logger.info(f"  Mean: {results['mean']} +/- {results['std']}")
    logger.info(f"  Range: [{results['min']}, {results['max']}]")
    logger.info(f"{'='*50}\n")

    return results


def get_feature_importance(model, feature_names: list,
                           top_n: int = 20) -> pd.DataFrame:
    """
    Extract and sort feature importances from a tree-based model.
    """
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    fi_df["importance_pct"] = (
        fi_df["importance"] / fi_df["importance"].sum() * 100
    ).round(2)

    return fi_df.reset_index(drop=True)


def backtest_predictions(actual: pd.Series, predicted: pd.Series,
                         dates: pd.Series, model_name: str = "Model") -> pd.DataFrame:
    """
    Create a backtest results DataFrame with actual vs predicted values
    aligned by date, plus error metrics per period.
    """
    bt = pd.DataFrame({
        "date": dates,
        "actual": actual.values,
        "predicted": predicted.values,
    })
    bt["error"] = bt["actual"] - bt["predicted"]
    bt["abs_error"] = bt["error"].abs()
    bt["pct_error"] = np.where(
        bt["actual"] != 0,
        (bt["error"] / bt["actual"] * 100).round(2),
        0.0,
    )
    bt["model"] = model_name
    return bt
