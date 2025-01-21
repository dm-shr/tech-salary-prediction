import logging
from typing import Any
from typing import Dict

import yaml


CONFIG_PATH = "configs/params.yaml"


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return config


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)
