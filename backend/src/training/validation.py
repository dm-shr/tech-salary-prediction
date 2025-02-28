"""
Model validation utilities for ensuring model quality before deployment.
"""
import logging
from typing import Any
from typing import Dict


class ModelValidationError(Exception):
    """Exception raised when model metrics don't meet required thresholds."""

    pass


def validate_model_scores(
    metrics: Dict[str, Any], thresholds: Dict[str, float], model_name: str
) -> bool:
    """
    Validate if model metrics meet the required thresholds.

    Args:
        metrics: Dictionary containing model evaluation metrics (r2_mean, mae_mean, rmse_mean)
        thresholds: Dictionary containing threshold values for r2, mae, rmse
        model_name: Name of the model being validated

    Returns:
        bool: True if all metrics meet thresholds

    Raises:
        ModelValidationError: If any metric doesn't meet the threshold
    """
    failed_metrics = []

    # Check R² (higher is better)
    if "r2" in thresholds and "r2_mean" in metrics:
        if metrics["r2_mean"] < thresholds["r2"]:
            failed_metrics.append(
                {"metric": "R²", "actual": float(metrics["r2_mean"]), "threshold": thresholds["r2"]}
            )

    # Check MAE (lower is better)
    if "mae" in thresholds and "mae_mean" in metrics:
        if metrics["mae_mean"] > thresholds["mae"] and thresholds["mae"] != float("inf"):
            failed_metrics.append(
                {
                    "metric": "MAE",
                    "actual": float(metrics["mae_mean"]),
                    "threshold": thresholds["mae"],
                }
            )

    # Check RMSE (lower is better)
    if "rmse" in thresholds and "rmse_mean" in metrics:
        if metrics["rmse_mean"] > thresholds["rmse"] and thresholds["rmse"] != float("inf"):
            failed_metrics.append(
                {
                    "metric": "RMSE",
                    "actual": float(metrics["rmse_mean"]),
                    "threshold": thresholds["rmse"],
                }
            )

    if failed_metrics:
        error_msg = f"{model_name} model failed validation checks:\n"
        for failure in failed_metrics:
            error_msg += f"- {failure['metric']}: {failure['actual']:.4f} (threshold: {failure['threshold']})\n"
        raise ModelValidationError(error_msg)

    return True


def log_validation_results(
    logger: logging.Logger,
    model_name: str,
    passed: bool,
    metrics: Dict[str, Any] = None,
    thresholds: Dict[str, float] = None,
) -> None:
    """
    Log the results of model validation.

    Args:
        logger: Logger instance
        model_name: Name of the model being validated
        passed: Whether validation passed
        metrics: Dictionary of model metrics (optional)
        thresholds: Dictionary of thresholds used (optional)
    """
    if passed:
        logger.info(f"{model_name} model passed validation checks")

        if metrics and thresholds:
            if "r2" in thresholds and "r2_mean" in metrics:
                logger.info(f"- R²: {metrics['r2_mean']:.4f} (threshold: {thresholds['r2']})")
            if "mae" in thresholds and "mae_mean" in metrics:
                logger.info(f"- MAE: {metrics['mae_mean']:.4f} (threshold: {thresholds['mae']})")
            if "rmse" in thresholds and "rmse_mean" in metrics:
                logger.info(f"- RMSE: {metrics['rmse_mean']:.4f} (threshold: {thresholds['rmse']})")
    else:
        logger.error(f"{model_name} model failed validation checks")
