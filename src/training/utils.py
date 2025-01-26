import gc
import subprocess

import mlflow
import numpy as np
import torch

from src.training.catboost.model import CatBoostModel
from src.training.transformer.model import SingleBERTWithMLP
from src.utils.utils import current_week_info
from src.utils.utils import load_config


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


def setup_mlflow(config):
    """Initialize MLflow tracking with config settings."""
    mlflow.set_tracking_uri(config["logging"]["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["logging"]["mlflow"]["experiment_name"])
    return mlflow


def load_models():
    """Load latest models from S3 using DVC"""
    config = load_config()

    # Get current week and year
    week_info = current_week_info()
    week = week_info["week_number"]
    year = week_info["year"]
    transformer_enabled = config["models"]["transformer"]["enabled"]
    models = {}

    # Construct model paths and pull from DVC
    model_configs = {
        "catboost": {
            "path": f"models/catboost/catboost_week_{week}_year_{year}.cbm",
            "loader": CatBoostModel.model_from_file,
            "enabled": True,
        },
        "transformer": {
            "path": f"models/transformer/transformer_week_{week}_year_{year}.pt",
            "loader": lambda path: SingleBERTWithMLP(config).load_state_dict(torch.load(path)),
            "enabled": transformer_enabled,
        },
    }

    # Load enabled models
    for model_name, config in model_configs.items():
        if config["enabled"]:
            subprocess.run(["dvc", "pull", config["path"]], check=True)
            models[model_name] = config["loader"](config["path"])

    return models
