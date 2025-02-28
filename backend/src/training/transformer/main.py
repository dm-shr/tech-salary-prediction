# transformer main.py
import logging
from typing import Any
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.training.transformer.dataset import split_tokenized_dict
from src.training.transformer.model import SingleBERTWithMLP
from src.training.transformer.train import fit_eval
from src.training.transformer.utils import log_metrics_to_mlflow
from src.training.transformer.utils import log_seed_metrics_and_plot
from src.training.transformer.utils import set_seed
from src.training.utils import setup_mlflow
from src.utils.utils import current_week_info  # dict with keys 'week_number' and 'year'
from src.utils.utils import load_config


config = load_config()
mlflow = setup_mlflow(config)


def main(logger: logging.Logger, enabled=True) -> Dict[str, Any]:
    # Get current week info for file naming
    week_info = current_week_info()
    week_suffix = f"week_{week_info['week_number']}_year_{week_info['year']}"

    # Early return with mock data if transformer is not enabled
    if not enabled:
        mock_array = np.array([0])
        y_pred_base = config["models"]["transformer"]["y_pred_base"]
        y_true_base = config["models"]["transformer"]["y_true_base"]
        save_base = config["models"]["transformer"]["save_base"]

        np.save(f"{y_pred_base}_{week_suffix}.npy", mock_array)
        np.save(f"{y_true_base}_{week_suffix}.npy", mock_array)
        torch.save({}, f"{save_base}_{week_suffix}.pt")

        return {"final": None}

    run_name = f"{config['models']['transformer']['mlflow_run_name']}_{week_suffix}"
    with mlflow.start_run(run_name=run_name) as _:
        # log seeds, test size
        mlflow.log_params(config["training"])

        # Load tokenized data with week suffix
        logger.info("Loading tokenized data...")
        tokenized_feature1_base = config["features"]["transformer"]["feature_processing"][0]["base"]
        tokenized_feature2_base = config["features"]["transformer"]["feature_processing"][1]["base"]
        tokenized_feature1 = torch.load(f"{tokenized_feature1_base}_{week_suffix}.pt")
        tokenized_feature2 = torch.load(f"{tokenized_feature2_base}_{week_suffix}.pt")

        # Load targets with week suffix
        targets = torch.load(f"{config['features']['target_base']}_{week_suffix}.pt")

        # get seeds
        main_seed = config["training"]["main_seed"]
        seeds = config["training"]["seeds"]

        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Loss function
        loss_function_name = config["models"]["transformer"]["loss_function"]
        if loss_function_name == "mse":
            criterion = nn.MSELoss()
        elif loss_function_name == "huber":
            criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function_name}")

        # Training and evaluation
        logger.info("Starting training and evaluation...")
        results = {}
        y_true = []
        y_pred = []

        # log model params
        model_name = (
            config["models"]["transformer"]["model_name"]
            if not config["is_test"]
            else config["models"]["transformer"]["model_name_test"]
        )
        hidden_size = (
            config["models"]["transformer"]["hidden_size"]
            if not config["is_test"]
            else config["models"]["transformer"]["hidden_size_test"]
        )
        mlflow.log_params(
            {
                "model_name": model_name,
                "hidden_size": hidden_size,
                "mlp_hidden_size": config["models"]["transformer"]["mlp_hidden_size"],
                "num_heads": config["models"]["transformer"]["num_heads"],
                "dropout": config["models"]["transformer"]["dropout"],
                "batch_size": config["models"]["transformer"]["batch_size"],
                "learning_rate": config["models"]["transformer"]["learning_rate"],
                "num_epochs": config["models"]["transformer"]["num_epochs"],
                "weight_decay": config["models"]["transformer"]["weight_decay"],
                "loss_function": config["models"]["transformer"]["loss_function"],
                "main_seed": config["training"]["main_seed"],
                "test_size": config["training"]["test_size"],
            }
        )

        # Initialize results dictionary
        results = {"final": None, "evaluation_best_epoch": None}

        # Training loop for each seed
        for i, seed in enumerate(seeds, 1):
            logger.info(f"Training model with seed {seed} ({i}/{len(seeds)})")
            set_seed(seed)

            # Initialize model
            model = SingleBERTWithMLP(config)
            model = torch.nn.DataParallel(model).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(config["models"]["transformer"]["learning_rate"]),
                weight_decay=float(config["models"]["transformer"]["weight_decay"]),
            )

            # Split data
            train_feature1, test_feature1 = split_tokenized_dict(
                tokenized_feature1, test_size=config["training"]["test_size"], random_state=seed
            )
            train_feature2, test_feature2 = split_tokenized_dict(
                tokenized_feature2, test_size=config["training"]["test_size"], random_state=seed
            )
            train_targets, test_targets = train_test_split(
                targets, test_size=config["training"]["test_size"], random_state=seed
            )

            train_data = {
                "tokenized_feature1": train_feature1,
                "tokenized_feature2": train_feature2,
                "targets": train_targets,
            }
            test_data = {
                "tokenized_feature1": test_feature1,
                "tokenized_feature2": test_feature2,
                "targets": test_targets,
            }

            model, history = fit_eval(
                model=model,
                train_data=train_data,
                test_data=test_data,
                criterion=criterion,
                optimizer=optimizer,
                config=config,
                device=device,
                logger=logger,
            )

            # Store results and log to MLflow
            y_pred.append(history["y_pred"])
            y_true.append(history["y_test"])
            results[seed] = history
            log_metrics_to_mlflow(history, str(seed))

        # Log aggregate metrics across seeds
        logger.info("Computing aggregate metrics across seeds...")
        metrics_summary = log_seed_metrics_and_plot(results, mlflow, logger)
        results["evaluation_best_epoch"] = metrics_summary

        # save y_pred and y_true as arrays with week suffix
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        y_pred_base = config["models"]["transformer"]["y_pred_base"]
        y_true_base = config["models"]["transformer"]["y_true_base"]
        np.save(f"{y_pred_base}_{week_suffix}.npy", y_pred)
        np.save(f"{y_true_base}_{week_suffix}.npy", y_true)

        # Train final model on full dataset
        logger.info("Training final model on complete dataset...")
        set_seed(main_seed)

        # Get data
        train_data = {
            "tokenized_feature1": tokenized_feature1,
            "tokenized_feature2": tokenized_feature2,
            "targets": targets,
        }
        final_model = SingleBERTWithMLP(config)
        final_model = torch.nn.DataParallel(final_model).to(device)
        final_model.train()
        model, history = fit_eval(
            model=final_model,
            train_data=train_data,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device=device,
            logger=logger,
            to_eval=False,
        )
        results["final"] = history
        log_metrics_to_mlflow(history, "final")

        # Save the model with week suffix
        save_path = f"{config['models']['transformer']['save_base']}_{week_suffix}.pt"
        torch.save(final_model.state_dict(), save_path)
        logger.info("Pipeline complete.")
        return results
