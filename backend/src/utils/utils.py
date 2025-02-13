import logging
from datetime import datetime
from typing import Any
from typing import Dict

import yaml


CONFIG_PATH = "configs/params.yaml"


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(name: str = __name__) -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)


def current_week_info():
    current_date = datetime.now()
    week_number = current_date.isocalendar()[1]
    year = current_date.year
    return {"week_number": week_number, "year": year}
