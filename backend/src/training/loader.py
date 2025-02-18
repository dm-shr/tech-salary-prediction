from src.training.catboost.model import CatBoostModel
from src.utils.s3_model_loader import S3ModelLoader
from src.utils.utils import load_config


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
        import torch
        from src.training.transformer.model import SingleBERTWithMLP

        transformer = SingleBERTWithMLP(config)
        transformer.load_state_dict(torch.load(model_paths["transformer"]))
        models["transformer"] = transformer

    return models
