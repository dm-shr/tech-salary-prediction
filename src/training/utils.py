import gc

import mlflow
import numpy as np
import torch

from src.training.catboost.model import CatBoostModel
from src.training.transformer.model import SingleBERTWithMLP
from src.utils.s3_model_loader import S3ModelLoader
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
    """Load latest models from S3"""
    config = load_config()
    s3_loader = S3ModelLoader()

    # Download latest models from S3
    model_paths = s3_loader.download_latest_models()
    models = {}

    # Load CatBoost model if available
    if model_paths.get("catboost"):
        models["catboost"] = CatBoostModel.model_from_file(model_paths["catboost"])

    # Load Transformer model if enabled and available
    if config["models"]["transformer"]["enabled"] and model_paths.get("transformer"):
        transformer = SingleBERTWithMLP(config)
        transformer.load_state_dict(torch.load(model_paths["transformer"]))
        models["transformer"] = transformer

    return models
