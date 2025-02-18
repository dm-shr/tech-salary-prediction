import logging

import numpy as np
from dotenv import load_dotenv

from src.training.catboost.main import main as train_catboost
from src.training.utils import setup_mlflow
from src.utils.s3_model_loader import S3ModelLoader
from src.utils.utils import current_week_info
from src.utils.utils import load_config
from src.utils.utils import setup_logging

load_dotenv(override=True)


def train_and_evaluate_single_model(
    logger: logging.Logger,
    config: dict,
    s3_loader: S3ModelLoader,
    week_info: dict,
    week_suffix: str,
) -> None:
    """Train and evaluate only CatBoost model when transformer is disabled."""
    logger.info("Training CatBoost model...")
    train_catboost(logger)

    # Upload CatBoost model
    catboost_local_path = f"{config['models']['catboost']['save_dir']}/catboost_{week_suffix}.cbm"
    s3_loader.upload_model("catboost", week_info, catboost_local_path)

    # Log CatBoost metrics (they're already logged in train_catboost)
    logger.info("CatBoost training and evaluation complete.")


def train_and_evaluate_blended(
    logger: logging.Logger,
    config: dict,
    s3_loader: S3ModelLoader,
    mlflow,
    week_info: dict,
    week_suffix: str,
) -> None:
    """Train and evaluate both models and blend their predictions."""
    # Import transformer-related modules conditionally
    from src.training.blend import blend_and_evaluate
    from src.training.transformer.main import main as train_transformer

    logger.info("Training both models for blending...")

    # Train CatBoost
    logger.info("Training CatBoost model...")
    train_catboost(logger)
    catboost_local_path = f"{config['models']['catboost']['save_dir']}/catboost_{week_suffix}.cbm"
    s3_loader.upload_model("catboost", week_info, catboost_local_path)

    # Train Transformer
    logger.info("Training Transformer model...")
    train_transformer(logger, enabled=True)
    transformer_local_path = (
        f"{config['models']['transformer']['save_base']}/transformer_{week_suffix}.pt"
    )
    s3_loader.upload_model("transformer", week_info, transformer_local_path)

    # Load predictions
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

    # Verify predictions alignment
    np.testing.assert_almost_equal(
        y_true_catboost,
        y_true_transformer[:, 0, :],
        decimal=4,
        err_msg="Mismatch in test set targets between models (beyond 4 decimal places).",
    )

    # Blend predictions
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

    # Log blended results to MLflow
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


def main(logger: logging.Logger):
    config = load_config()
    mlflow = setup_mlflow(config)
    s3_loader = S3ModelLoader()

    # Get current week info for file naming
    week_info = current_week_info()
    week_suffix = f"week_{week_info['week_number']}_year_{week_info['year']}"

    if not config["models"]["transformer"]["enabled"]:
        logger.info("Transformer is disabled. Running CatBoost-only training...")
        train_and_evaluate_single_model(logger, config, s3_loader, week_info, week_suffix)
    else:
        logger.info("Running full model training with blending...")
        train_and_evaluate_blended(logger, config, s3_loader, mlflow, week_info, week_suffix)

    logger.info("Training complete.")


if __name__ == "__main__":
    logger = setup_logging()
    main(logger)
