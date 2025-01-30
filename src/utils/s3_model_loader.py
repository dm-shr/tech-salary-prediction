import logging
import os
from pathlib import Path
from typing import Dict
from typing import Optional

import boto3
from dotenv import load_dotenv

from src.utils.utils import current_week_info
from src.utils.utils import load_config

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3ModelLoader:
    def __init__(
        self,
        # local_model_dir='/app/models',
        local_model_dir="models",
        endpoint_url: Optional[str] = None,
    ):
        """
        Initialize S3 model loader

        Args:
            bucket_name: Name of the S3 bucket
            local_model_dir: Local directory to save downloaded models
            endpoint_url: Optional custom S3 endpoint URL (for MinIO, etc.)
        """
        self.bucket_name = os.getenv("AWS_BUCKET_NAME")
        self.local_model_dir = Path(local_model_dir)
        self.config = load_config()

        # Initialize model types from config
        self.MODEL_TYPES = {
            "catboost": {
                "prefix": self.config["models"]["catboost"]["save_dir"],
                "pattern": "catboost_week_{week}_year_{year}.cbm",
                "enabled": True,  # CatBoost is always enabled in config
            },
            "transformer": {
                "prefix": self.config["models"]["transformer"]["save_base"],
                "pattern": "transformer_week_{week}_year_{year}.pt",
                "enabled": self.config["models"]["transformer"]["enabled"],
            },
        }

        # Initialize S3 client
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION"),
        )

        # Create local directory if it doesn't exist
        self.local_model_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_type: str, week_info: Dict[str, int]) -> Dict[str, str]:
        """
        Generate S3 and local paths for a specific model version

        Args:
            model_type: Type of model ('catboost' or 'transformer')
            week_info: Dictionary containing 'week_number' and 'year'

        Returns:
            Dict with 's3_path' and 'local_path'
        """
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Unknown model type: {model_type}")

        model_config = self.MODEL_TYPES[model_type]
        if not model_config["enabled"]:
            logger.info(f"Model type {model_type} is disabled in config")
            return None

        filename = model_config["pattern"].format(
            week=week_info["week_number"], year=week_info["year"]
        )

        s3_path = f"{model_config['prefix']}/{filename}"
        local_path = self.local_model_dir / model_type / filename

        return {"s3_path": s3_path, "local_path": local_path}

    def download_latest_models(self) -> Dict[str, Optional[str]]:
        """
        Download the latest version of each enabled model type based on current week

        Returns:
            Dict mapping model type to downloaded model filename or None if failed
        """
        week_info = current_week_info()
        results = {}

        for model_type in self.MODEL_TYPES:
            if not self.MODEL_TYPES[model_type]["enabled"]:
                logger.info(f"Skipping disabled model type: {model_type}")
                continue

            try:
                paths = self.get_model_path(model_type, week_info)
                if not paths:
                    continue

                local_path = paths["local_path"]
                s3_path = paths["s3_path"]

                # Create subdirectories if needed
                local_path.parent.mkdir(parents=True, exist_ok=True)

                # Download the file
                logger.info(f"Downloading {model_type} model for week {week_info['week_number']}")
                self.s3.download_file(self.bucket_name, s3_path, str(local_path))
                logger.info(f"Successfully downloaded {model_type} model")
                results[model_type] = str(local_path)

            except self.s3.exceptions.NoSuchKey:
                logger.warning(f"No {model_type} model found for week {week_info['week_number']}")
                results[model_type] = None
            except Exception as e:
                logger.error(f"Error downloading {model_type} model: {str(e)}")
                results[model_type] = None

        return results

    def upload_model(self, model_type: str, week_info: Dict[str, int], local_path: str) -> bool:
        """
        Upload a model file to S3

        Args:
            model_type: Type of model ('catboost' or 'transformer')
            week_info: Dictionary containing 'week_number' and 'year'
            local_path: Path to the local model file

        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            paths = self.get_model_path(model_type, week_info)
            if not paths:
                return False

            s3_path = paths["s3_path"]
            self.s3.upload_file(local_path, self.bucket_name, s3_path)
            logger.info(f"Successfully uploaded {model_type} model to S3")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {model_type} model to S3: {str(e)}")
            return False

    def cleanup_old_models(self):
        """Remove all models except the current versions"""
        week_info = current_week_info()

        for model_type in self.MODEL_TYPES:
            if not self.MODEL_TYPES[model_type]["enabled"]:
                continue

            model_dir = self.local_model_dir / model_type
            if not model_dir.exists():
                continue

            paths = self.get_model_path(model_type, week_info)
            if not paths:
                continue

            current_model = paths["local_path"]

            for model_file in model_dir.glob("*"):
                if model_file != current_model and model_file.is_file():
                    logger.info(f"Removing old model: {model_file}")
                    model_file.unlink()
