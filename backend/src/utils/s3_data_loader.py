import logging
import os
from pathlib import Path
from typing import Dict
from typing import Optional

import boto3
from dotenv import load_dotenv

from src.utils.utils import load_config

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3DataLoader:
    def __init__(self, local_data_dir: str = "data", endpoint_url: Optional[str] = None):
        """
        Initialize S3 data loader

        Args:
            local_data_dir: Local directory to save downloaded data
            endpoint_url: Optional custom S3 endpoint URL (for MinIO, etc.)
        """
        self.bucket_name = os.getenv("AWS_BUCKET_NAME")
        self.local_data_dir = Path(local_data_dir)
        self.config = load_config()

        # Initialize data types from config
        self.DATA_TYPES = {
            "getmatch": {
                "prefix": "data/raw/getmatch",
                "pattern": f"{self.config['preprocessing']['input_filename_base']['getmatch']}_week_{{week}}_year_{{year}}.csv",
            },
            "headhunter": {
                "prefix": "data/raw/headhunter",
                "pattern": f"{self.config['preprocessing']['input_filename_base']['headhunter']}_week_{{week}}_year_{{year}}.csv",
            },
            "historical": {
                "prefix": "data/preprocessed/historical",
                "filename": self.config["preprocessing"]["historical_data_path"],
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

        # Create local directories
        self.local_data_dir.mkdir(parents=True, exist_ok=True)
        (self.local_data_dir / "raw").mkdir(exist_ok=True)
        (self.local_data_dir / "processed").mkdir(exist_ok=True)

    def get_data_path(
        self, data_type: str, week_info: Optional[Dict[str, int]] = None
    ) -> Dict[str, str]:
        """Generate S3 and local paths for a specific data file"""
        if data_type not in self.DATA_TYPES:
            raise ValueError(f"Unknown data type: {data_type}")

        data_config = self.DATA_TYPES[data_type]

        if data_type == "historical":
            filename = Path(data_config["filename"]).name
            s3_path = f"{data_config['prefix']}/{filename}"
            local_path = self.local_data_dir / "processed" / filename
        else:
            if week_info is None:
                raise ValueError(f"week_info required for data type: {data_type}")
            filename = data_config["pattern"].format(
                week=week_info["week_number"], year=week_info["year"]
            )
            s3_path = f"{data_config['prefix']}/{filename}"
            local_path = self.local_data_dir / "raw" / filename

        return {"s3_path": s3_path, "local_path": local_path}

    def download_data(
        self, data_type: str, week_info: Optional[Dict[str, int]] = None
    ) -> Optional[str]:
        """Download a specific data file from S3"""
        try:
            paths = self.get_data_path(data_type, week_info)
            local_path = paths["local_path"]
            s3_path = paths["s3_path"]

            # Create subdirectories if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download the file
            logger.info(f"Downloading {data_type} data...")
            self.s3.download_file(self.bucket_name, s3_path, str(local_path))
            logger.info(f"Successfully downloaded {data_type} data")
            return str(local_path)

        except self.s3.exceptions.NoSuchKey:
            logger.warning(f"No {data_type} data found in S3")
            return None
        except Exception as e:
            logger.error(f"Error downloading {data_type} data: {str(e)}")
            return None

    def upload_data(
        self, data_type: str, local_path: str, week_info: Optional[Dict[str, int]] = None
    ) -> bool:
        """Upload a data file to S3"""
        try:
            paths = self.get_data_path(data_type, week_info)
            s3_path = paths["s3_path"]

            self.s3.upload_file(local_path, self.bucket_name, s3_path)
            logger.info(f"Successfully uploaded {data_type} data to S3")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {data_type} data to S3: {str(e)}")
            return False
