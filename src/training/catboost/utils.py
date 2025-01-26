import os
import pickle
import tempfile
from typing import Any
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool
from scipy.stats import t

from src.utils.utils import current_week_info  # dict with keys 'week_number' and 'year'
from src.utils.utils import load_config


config = load_config()


def save_history(history: Dict[int, Dict[str, Any]], save_dir: str, model_name: str) -> None:
    """
    Save training history to a pickle file.

    Args:
        history: Dictionary containing training history
        save_dir: Directory to save the history
        model_name: Name of the model for the file
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_history.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(history, f)


def create_pool_data(
    config: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series = None,
):
    """
    Create CatBoost Pool object from data.

    Args:
        config: Configuration dictionary containing feature definitions
        X: Features DataFrame
        y: Target values

    Returns:
        CatBoost Pool object
    """
    text_features = config["features"]["features"]["catboost"]["text"]
    cat_features = config["features"]["features"]["catboost"]["categorical"]
    return Pool(
        data=X,
        label=y,
        text_features=text_features,
        cat_features=cat_features,
    )


def load_data(config):
    """
    Load preprocessed data based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Features and target data
    """
    # Get current week info
    week_info = current_week_info()
    week_suffix = f"week_{week_info['week_number']}_year_{week_info['year']}"

    # Load preprocessed data
    data_path = config["features"]["catboost"]["features_base"] + f"_{week_suffix}.csv"
    target_path = config["features"]["target_base"] + f"_{week_suffix}.csv"

    data = pd.read_csv(data_path)
    target = pd.read_csv(target_path)

    # Get feature lists
    text_features = config["features"]["features"]["catboost"]["text"]
    cat_features = config["features"]["features"]["catboost"]["categorical"]
    num_features = config["features"]["features"]["catboost"]["numeric"]

    # Combine all features
    features = text_features + cat_features + num_features
    target_col = config["features"]["target_name"]

    X = data[features]
    y = target[target_col]

    return X, y


def display_metrics_with_ci(history: dict, logger, mlflow):
    """
    Plot mean and confidence intervals for train and test R² metrics.
    Log the metrics and CI to MLflow.
    """
    seeds = list(history.keys())
    alpha = config["training"]["confidence_interval"]["alpha"]

    def calculate_ci(data):
        n = len(data)
        if n == 0:
            return np.nan, np.nan, np.nan
        # m, se = np.nanmean(data), np.nanstd(data) / np.sqrt(n)
        m, se = np.mean(data), np.std(data) / np.sqrt(n)
        t_value = t.ppf(1 - (alpha / 2), n - 1)
        h = se * t_value
        return m, m - h, m + h

    max_len = max(len(history[seed]["r2_train"]) for seed in seeds)
    r2_train_values = np.array(
        [
            np.pad(
                history[seed]["r2_train"],
                (0, max_len - len(history[seed]["r2_train"])),
                mode="constant",
                constant_values=np.nan,
            )
            for seed in seeds
        ]
    )
    r2_test_values = np.array(
        [
            np.pad(
                history[seed]["r2_test"],
                (0, max_len - len(history[seed]["r2_test"])),
                mode="constant",
                constant_values=np.nan,
            )
            for seed in seeds
        ]
    )

    r2_train_mean = np.mean(r2_train_values, axis=0)
    r2_test_mean = np.mean(r2_test_values, axis=0)

    # r2_train_mean = np.nanmean(r2_train_values, axis=0)
    # r2_test_mean = np.nanmean(r2_test_values, axis=0)

    r2_train_ci = np.array(
        [calculate_ci(r2_train_values[:, i]) for i in range(r2_train_values.shape[1])]
    )
    r2_test_ci = np.array(
        [calculate_ci(r2_test_values[:, i]) for i in range(r2_test_values.shape[1])]
    )

    plt.figure(figsize=(10, 6))
    plt.plot(r2_train_mean, label="Train R²")
    plt.fill_between(range(len(r2_train_mean)), r2_train_ci[:, 1], r2_train_ci[:, 2], alpha=0.3)
    plt.plot(r2_test_mean, label="Test R²")
    plt.fill_between(range(len(r2_test_mean)), r2_test_ci[:, 1], r2_test_ci[:, 2], alpha=0.3)
    plt.title("Mean R² by Iteration with 95% CI")
    plt.xlabel("Iteration")
    plt.ylabel("R²")
    plt.legend()
    plt.grid(True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name)
        mlflow.log_artifact(tmp.name, "plots/mean_r2_ci.png")
        os.unlink(tmp.name)

    r2_values = [history[seed]["r2"] for seed in seeds]
    mae_values = [history[seed]["mae"] for seed in seeds]
    rmse_values = [history[seed]["rmse"] for seed in seeds]

    metrics_summary = {}

    for metric_name, values in zip(["R²", "MAE", "RMSE"], [r2_values, mae_values, rmse_values]):
        mean, lower, upper = calculate_ci(values)
        metrics_summary.update(
            {
                f"{metric_name.lower()}_mean": mean,
                f"{metric_name.lower()}_ci_lower": lower,
                f"{metric_name.lower()}_ci_upper": upper,
            }
        )
        logger.info(f"{metric_name}: Mean = {mean:.4f}, 95% CI = [{lower:.4f}, {upper:.4f}]")

    mlflow.log_metrics(metrics_summary)
