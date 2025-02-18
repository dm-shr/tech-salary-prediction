from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.fastapi_app import app
from src.training.catboost.model import CatBoostModel


@pytest.fixture
def mock_config_transformer_enabled():
    return {
        "models": {
            "transformer": {"enabled": True},
            "blended": {"catboost_weight": 0.5, "transformer_weight": 0.5},
        },
        "features": {
            "transformer": {
                "add_query_prefix": True,
                "feature_processing": [
                    {"name": "description_no_numbers", "max_len": 512},
                    {"name": "title_company_location_skills_source", "max_len": 256},
                ],
            }
        },
    }


@pytest.fixture
def mock_config_transformer_disabled():
    return {
        "models": {
            "transformer": {"enabled": False},
            "blended": {"catboost_weight": 1.0, "transformer_weight": 0.0},
        }
    }


@pytest.fixture
def mock_catboost_model():
    model = Mock(spec=CatBoostModel)
    model.predict.return_value = np.array([10.0])  # log-scale prediction
    return model


@pytest.fixture
def mock_transformer_model():
    model = Mock()
    model.predict.return_value = np.array([10.0])  # log-scale prediction
    return model


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def sample_input():
    return {
        "title": "Data Scientist",
        "company": "Test Company",
        "location": "Stockholm",
        "description": "Looking for a Data Scientist",
        "skills": "python,sql",
        "experience_from": 2,
        "experience_to": 5,
    }
