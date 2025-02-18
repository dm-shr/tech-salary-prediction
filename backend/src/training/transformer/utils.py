import gc

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from scipy.stats import t


def memory_cleanup():
    """Clean up memory."""
    gc.collect()
    torch.cuda.empty_cache()


def set_seed(seed: int) -> None:
    """Set seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    m, se = np.mean(data), np.std(data) / np.sqrt(n)
    h = se * t.ppf((1 + confidence) / 2, n - 1)
    return m, m - h, m + h


def log_seed_metrics_and_plot(results, mlflow, logger):
    """Calculate statistics across seeds and create visualizations."""
    # Filter out 'final' from results
    seed_keys = [k for k in results.keys() if k != "final"]

    # Prepare arrays for each metric
    r2_train_values = np.array([results[seed]["train_r2"] for seed in seed_keys])
    r2_test_values = np.array([results[seed]["test_r2"] for seed in seed_keys])
    mae_test_values = np.array([results[seed]["test_mae"] for seed in seed_keys])
    rmse_test_values = np.array([results[seed]["test_rmse"] for seed in seed_keys])

    # Calculate means
    r2_train_mean = np.mean(r2_train_values, axis=0)
    r2_test_mean = np.mean(r2_test_values, axis=0)
    mae_test_mean = np.mean(mae_test_values, axis=0)
    rmse_test_mean = np.mean(rmse_test_values, axis=0)

    # Calculate confidence intervals
    r2_train_ci = np.array(
        [mean_confidence_interval(r2_train_values[:, i]) for i in range(r2_train_values.shape[1])]
    )
    r2_test_ci = np.array(
        [mean_confidence_interval(r2_test_values[:, i]) for i in range(r2_test_values.shape[1])]
    )
    mae_test_ci = np.array(
        [mean_confidence_interval(mae_test_values[:, i]) for i in range(mae_test_values.shape[1])]
    )
    rmse_test_ci = np.array(
        [mean_confidence_interval(rmse_test_values[:, i]) for i in range(rmse_test_values.shape[1])]
    )

    # Find best epoch based on test R2
    best_epoch = np.argmax(r2_test_mean)

    # Log best epoch metrics with confidence intervals
    best_metrics = {
        "best_epoch": best_epoch + 1,
        "best_r2_test_mean": r2_test_mean[best_epoch],
        "best_r2_test_ci_lower": r2_test_ci[best_epoch, 1],
        "best_r2_test_ci_upper": r2_test_ci[best_epoch, 2],
        "best_mae_test_mean": mae_test_mean[best_epoch],
        "best_mae_test_ci_lower": mae_test_ci[best_epoch, 1],
        "best_mae_test_ci_upper": mae_test_ci[best_epoch, 2],
        "best_rmse_test_mean": rmse_test_mean[best_epoch],
        "best_rmse_test_ci_lower": rmse_test_ci[best_epoch, 1],
        "best_rmse_test_ci_upper": rmse_test_ci[best_epoch, 2],
    }
    mlflow.log_metrics(best_metrics)

    # Create and save the plot
    plt.figure(figsize=(10, 6))
    plt.plot(r2_train_mean[1:], label="train")
    plt.fill_between(
        range(len(r2_train_mean[1:])), r2_train_ci[1:, 1], r2_train_ci[1:, 2], alpha=0.3
    )
    plt.plot(r2_test_mean[1:], label="test")
    plt.fill_between(range(len(r2_test_mean[1:])), r2_test_ci[1:, 1], r2_test_ci[1:, 2], alpha=0.3)
    plt.title("Mean R2 by epoch, with 95% CI")
    plt.xlabel("Epoch")
    plt.ylabel("R2")
    plt.legend()

    # Save plot to MLflow
    mlflow.log_figure(plt.gcf(), "r2_confidence_intervals.png")
    plt.close()

    # Log results using logger
    logger.info(f"TEST METRICS FOR THE BEST EPOCH (#{best_epoch+1})")
    logger.info(
        f"R2: mean = {r2_test_mean[best_epoch]:.4f}, "
        f"95% CI = [{r2_test_ci[best_epoch, 1]:.4f}, {r2_test_ci[best_epoch, 2]:.4f}]"
    )
    logger.info(
        f"MAE: mean = {mae_test_mean[best_epoch]:.4f}, "
        f"95% CI = [{mae_test_ci[best_epoch, 1]:.4f}, {mae_test_ci[best_epoch, 2]:.4f}]"
    )
    logger.info(
        f"RMSE: mean = {rmse_test_mean[best_epoch]:.4f}, "
        f"95% CI = [{rmse_test_ci[best_epoch, 1]:.4f}, {rmse_test_ci[best_epoch, 2]:.4f}]"
    )


def log_metrics_to_mlflow(history, seed_name):
    """Log metrics for each epoch to MLflow."""
    n_epochs = len(history["train_loss"])
    for epoch in range(n_epochs):
        metrics = {
            f"{seed_name}/train_loss": history["train_loss"][epoch],
            f"{seed_name}/train_rmse": history["train_rmse"][epoch],
            f"{seed_name}/train_r2": history["train_r2"][epoch],
            f"{seed_name}/train_mae": history["train_mae"][epoch],
        }

        # Add test metrics if they exist
        if "test_loss" in history and history["test_loss"]:
            metrics.update(
                {
                    f"{seed_name}/test_loss": history["test_loss"][epoch],
                    f"{seed_name}/test_rmse": history["test_rmse"][epoch],
                    f"{seed_name}/test_r2": history["test_r2"][epoch],
                    f"{seed_name}/test_mae": history["test_mae"][epoch],
                }
            )

        mlflow.log_metrics(metrics, step=epoch)
