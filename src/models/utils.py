import torch
import numpy as np
import gc
import mlflow


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
