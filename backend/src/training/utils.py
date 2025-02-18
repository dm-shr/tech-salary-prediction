import mlflow


def setup_mlflow(config):
    """Initialize MLflow tracking with config settings."""
    mlflow.set_tracking_uri(config["logging"]["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["logging"]["mlflow"]["experiment_name"])
    return mlflow
