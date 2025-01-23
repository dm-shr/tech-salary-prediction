import logging

import numpy as np

from src.training.blend import blend_and_evaluate
from src.training.catboost.main import main as train_catboost
from src.training.transformer.main import main as train_transformer
from src.training.utils import setup_mlflow
from src.utils.utils import current_week_info  # dict with keys 'week_number' and 'year'
from src.utils.utils import load_config


def main(logger: logging.Logger):
    config = load_config()
    mlflow = setup_mlflow(config)

    # Get current week info for file naming
    week_info = current_week_info()
    week_suffix = f"week_{week_info['week_number']}_year_{week_info['year']}"

    logger.info("Starting blended model training...")

    # Train CatBoost Model
    logger.info("Training CatBoost model...")
    train_catboost(logger)

    if not config["models"]["transformer"]["enabled"]:
        logger.info("Transformer is disabled. Saving mock data...")
        train_transformer(
            logger,
            enabled=False,
        )  # this will save mock data as config has transformer disabled
        logger.info("Training complete.")
        return

    # Train Transformer Model
    logger.info("Training Transformer model...")
    train_transformer(logger)

    # Load predictions and true values with week suffix
    catboost_predictions = np.load(
        f"{config['models']['catboost']['y_pred_base']}_{week_suffix}.npy"
    )
    transformer_predictions = np.load(
        f"{config['models']['transformer']['y_pred_base']}_{week_suffix}.npy"
    )
    y_true_catboost = np.load(f"{config['models']['catboost']['y_true_base']}_{week_suffix}.npy")
    y_true_transformer = np.load(
        f"{config['models']['transformer']['y_true_base']}_{week_suffix}.npy"
    )

    # Assert that true values match between models (allowing for small numerical differences)
    np.testing.assert_almost_equal(
        y_true_catboost,
        y_true_transformer[:, 0, :],
        decimal=4,
        err_msg="Mismatch in test set targets between models (beyond 4 decimal places).",
    )

    # Get blend weights from config or use default
    catboost_weight = config["models"]["blended"]["catboost_weight"]
    transformer_weight = config["models"]["blended"]["transformer_weight"]

    # Calculate blended metrics
    logger.info("Calculating blended metrics...")
    results = blend_and_evaluate(
        catboost_preds=catboost_predictions,
        transformer_preds=transformer_predictions,
        y_true=y_true_catboost,
        blend_weights=(catboost_weight, transformer_weight),
        alpha=config["training"]["confidence_interval"]["alpha"],
    )

    # Log metrics and parameters to MLflow
    run_name = f"{config['models']['blended']['mlflow_run_name']}_{week_suffix}"

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("best_epoch", results["best_epoch"])
        mlflow.log_param("blend_weight_catboost", catboost_weight)
        mlflow.log_param("blend_weight_transformer", transformer_weight)

        # Log metrics with confidence intervals
        for metric_name, values in results["metrics"].items():
            mlflow.log_metric(f"best_{metric_name}_mean", values[0])
            mlflow.log_metric(f"best_{metric_name}_ci_lower", values[1])
            mlflow.log_metric(f"best_{metric_name}_ci_upper", values[2])

        # Log results
        logger.info(f"Best epoch: {results['best_epoch']}")
        for metric_name, values in results["metrics"].items():
            logger.info(
                f"{metric_name.upper()}: mean={values[0]:.4f}, "
                f"CI=[{values[1]:.4f}, {values[2]:.4f}]"
            )

    logger.info("Blended model evaluation complete.")
    logger.info("Training complete.")
