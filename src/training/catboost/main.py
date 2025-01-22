# catboost main.py
import numpy as np
from sklearn.model_selection import train_test_split

from src.training.catboost.model import get_model_config
from src.training.catboost.model import save_model
from src.training.catboost.model import train_and_evaluate_with_metrics
from src.training.catboost.utils import create_pool_data
from src.training.catboost.utils import display_metrics_with_ci
from src.training.catboost.utils import load_data
from src.training.catboost.utils import save_history
from src.training.utils import setup_mlflow
from src.utils.utils import current_week_info  # dict with keys 'week_number' and 'year'
from src.utils.utils import load_config
from src.utils.utils import setup_logging


def main():
    # get configuration and logging
    config = load_config()
    mlflow = setup_mlflow(config)
    logger = setup_logging()

    # Get current week info for file naming
    week_info = current_week_info()
    week_suffix = f"week_{week_info['week_number']}_year_{week_info['year']}"

    run_name = f"{config['models']['catboost']['mlflow_run_name']}_{week_suffix}"
    with mlflow.start_run(run_name=run_name) as _:
        # log seeds, test size
        mlflow.log_params(config["training"])
        # get the data
        logger.info("Loading data...")
        X, y = load_data(config)

        seeds = config["training"]["seeds"]
        main_seed = config["training"]["main_seed"]
        test_size = config["training"]["test_size"]
        # dict to store performance metrics
        history = {}
        y_true_combined = []
        y_pred_combined = []

        # Log model parameters (using first seed)
        model = get_model_config(config, seeds[0])
        model_params = model.get_params()
        mlflow.log_params({f"model_{key}": value for key, value in model_params.items()})

        # train and evaluate with different seeds
        for seed in seeds:
            logger.info(f"Running with seed = {seed}")
            # prepare data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed
            )
            # print data shapes
            logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

            train_pool = create_pool_data(X_train, y_train, config)
            test_pool = create_pool_data(X_test, y_test, config)

            # get model
            model = get_model_config(config, seed)

            # train and evaluate
            metrics = train_and_evaluate_with_metrics(
                model=model,
                train_pool=train_pool,
                test_pool=test_pool,
                X_test=X_test,
                y_test=y_test,
            )

            # store metrics per seed
            history[seed] = metrics
            y_true_combined.append(metrics["y_test"])
            y_pred_combined.append(metrics["y_pred"])

        logger.info("Training-evaluation complete. displaying metrics...")
        # Log metrics with CI and the R2 plot
        display_metrics_with_ci(history, logger, mlflow)
        # Save history of the performance metrics with week suffix
        logger.info("Saving history of the performance metrics...")
        save_history(
            history,
            config["models"]["catboost"]["save_dir"],
            f"catboost_{week_suffix}",
        )

        # save y_true and y_pred with week suffix
        logger.info("Saving predictions")
        y_true_combined = np.array(y_true_combined)
        y_pred_combined = np.array(y_pred_combined)
        y_true_base = config["models"]["catboost"]["y_true_base"]
        y_pred_base = config["models"]["catboost"]["y_pred_base"]
        np.save(f"{y_true_base}_{week_suffix}.npy", y_true_combined)
        np.save(f"{y_pred_base}_{week_suffix}.npy", y_pred_combined)

        # Train model on the entire dataset with main_seed
        logger.info("\nTraining final model on the full dataset...")
        final_model = get_model_config(config, seed=main_seed)

        # Create data pool for the entire dataset
        full_pool = create_pool_data(X, y, config)

        # Train the final model, calculate train metrics for the full dataset
        final_train_metrics = train_and_evaluate_with_metrics(
            model=final_model,
            train_pool=full_pool,
            X_train=X,
            y_train=y,
            to_eval=False,
        )

        # Log final training metrics to MLflow
        mlflow.log_metrics(
            {
                f"final_train_{metric_name}": value
                for metric_name, value in final_train_metrics.items()
            }
        )

        logger.info("Final model metrics (trained on full dataset):")
        for metric_name, value in final_train_metrics.items():
            logger.info(f"{metric_name.upper()} = {value:.4f}")

        # Save final model with week suffix
        save_model(model, config, f"catboost_{week_suffix}")

        logger.info("Training complete.")


if __name__ == "__main__":
    main()
