from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
from scipy.stats import t
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def mean_confidence_interval(data, alpha=0.05):
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan
    m, se = np.mean(data), np.std(data) / np.sqrt(n)
    t_value = t.ppf(1 - (alpha / 2), n - 1)
    h = se * t_value
    return m, m - h, m + h


def calculate_metrics(y_true, y_pred):
    """Calculate R2, MAE, and RMSE for given predictions."""
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse


def blend_and_evaluate(
    catboost_preds, transformer_preds, y_true, blend_weights=(0.5, 0.5), alpha=0.05
) -> Tuple[Dict[str, Union[int, Dict[str, Tuple[float, float, float]]]], Dict[str, float]]:
    """
    Blend predictions and evaluate metrics.

    Args:
        catboost_preds: shape (n_seeds, n_samples)
        transformer_preds: shape (n_seeds, n_epochs, n_samples)
        y_true: shape (n_seeds, n_samples)
        blend_weights: tuple of (catboost_weight, transformer_weight)
        alpha: confidence level for confidence intervals, default 0.05

    Returns:
        dict: Dictionary containing metrics and best epoch information
    """
    n_seeds, n_epochs, n_samples = transformer_preds.shape
    w1, w2 = blend_weights

    # Initialize arrays to store metrics for each seed and epoch
    r2_scores = np.zeros((n_seeds, n_epochs))
    mae_scores = np.zeros((n_seeds, n_epochs))
    rmse_scores = np.zeros((n_seeds, n_epochs))

    # Calculate metrics for each seed and epoch
    for seed in range(n_seeds):
        # Expand catboost predictions to match transformer shape
        catboost_expanded = np.expand_dims(catboost_preds[seed], axis=0)
        catboost_expanded = np.repeat(catboost_expanded, n_epochs, axis=0)

        # Blend predictions for all epochs
        blended_preds = w1 * catboost_expanded + w2 * transformer_preds[seed]

        # Calculate metrics for each epoch
        for epoch in range(n_epochs):
            r2, mae, rmse = calculate_metrics(y_true[seed], blended_preds[epoch])
            r2_scores[seed, epoch] = r2
            mae_scores[seed, epoch] = mae
            rmse_scores[seed, epoch] = rmse

    # Calculate mean metrics across seeds for each epoch
    mean_r2 = np.mean(r2_scores, axis=0)
    mean_mae = np.mean(mae_scores, axis=0)
    mean_rmse = np.mean(rmse_scores, axis=0)

    # Find best epoch based on R2 score
    best_epoch = np.argmax(mean_r2)

    # Calculate confidence intervals for best epoch
    best_r2_ci = mean_confidence_interval(r2_scores[:, best_epoch], alpha)
    best_mae_ci = mean_confidence_interval(mae_scores[:, best_epoch], alpha)
    best_rmse_ci = mean_confidence_interval(rmse_scores[:, best_epoch], alpha)

    results_for_logging = {
        "best_epoch": best_epoch + 1,
        "metrics": {
            "r2": best_r2_ci,
            "mae": best_mae_ci,
            "rmse": best_rmse_ci,
        },
        "all_metrics": {
            "r2": mean_r2,
            "mae": mean_mae,
            "rmse": mean_rmse,
        },
    }

    results_for_testing = {
        "r2_mean": best_r2_ci[0],
        "r2_ci_lower": best_r2_ci[1],
        "r2_ci_upper": best_r2_ci[2],
        "mae_mean": best_mae_ci[0],
        "mae_ci_lower": best_mae_ci[1],
        "mae_ci_upper": best_mae_ci[2],
        "rmse_mean": best_rmse_ci[0],
        "rmse_ci_lower": best_rmse_ci[1],
        "rmse_ci_upper": best_rmse_ci[2],
    }

    return results_for_logging, results_for_testing
