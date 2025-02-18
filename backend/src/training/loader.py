import logging

from src.training.catboost.model import CatBoostModel
from src.utils.s3_model_loader import S3ModelLoader
from src.utils.utils import load_config

logger = logging.getLogger(__name__)


def load_models():
    """Load latest models from S3"""
    config = load_config()
    s3_loader = S3ModelLoader()

    # Download latest models from S3
    model_paths = s3_loader.download_latest_models()
    models = {}

    # Load CatBoost model if available
    if model_paths.get("catboost"):
        catboost_path = model_paths["catboost"]
        logger.info(f"Attempting to load CatBoost model from: {catboost_path}")
        try:
            models["catboost"] = CatBoostModel.model_from_file(catboost_path)
            logger.info("CatBoost model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading CatBoost model from path: {catboost_path}. Exception: {e}")
            raise
    else:
        logger.warning("No CatBoost model path found. Skipping CatBoost model loading.")

    # Load Transformer model if enabled and available
    if config["models"]["transformer"]["enabled"] and model_paths.get("transformer"):
        transformer_path = model_paths["transformer"]
        logger.info(f"Attempting to load Transformer model from: {transformer_path}")
        try:
            import torch
            from src.training.transformer.model import SingleBERTWithMLP

            transformer = SingleBERTWithMLP(config)
            transformer.load_state_dict(torch.load(transformer_path))
            models["transformer"] = transformer
            logger.info("Transformer model loaded successfully.")
        except Exception as e:
            logger.error(
                f"Error loading Transformer model from path: {transformer_path}. Exception: {e}"
            )
            raise
    elif config["models"]["transformer"]["enabled"]:
        logger.warning("Transformer model is enabled but no model path found.  Check S3 or DVC.")
    else:
        logger.info("Transformer model loading skipped (disabled in config).")

    return models
