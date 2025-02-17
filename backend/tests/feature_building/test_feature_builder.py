import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.feature_building.main import FeatureBuilder


@pytest.fixture
def mock_config():
    return {
        "is_test": True,
        "models": {
            "transformer": {
                "enabled": True,
            },
        },
        "features": {
            "preprocessed_data_base": "dummy_path",
            "output_base": "dummy_path",
            "target_base": "dummy_path",
            "target_name": "log_salary_from",
            "test_size": 1.0,
            "features": {
                "catboost": {
                    "text": ["title", "location", "company", "description_no_numbers_with_skills"],
                    "categorical": ["source"],
                    "numeric": ["experience_from", "experience_to_adjusted_10", "description_size"],
                },
                "transformer": {
                    "text": ["description_no_numbers", "title_company_location_skills_source"]
                },
                "bi_gru_cnn": {
                    "text": ["description_no_numbers", "title_company_location_skills_source"]
                },
            },
            "transformer": {
                "features_base": "dummy_path",
                "tokenizer": "intfloat/multilingual-e5-small",
                "tokenizer_test": "sergeyzh/rubert-tiny-turbo",
                "add_query_prefix": True,
                "feature_processing": [
                    {"name": "description_no_numbers", "max_len": 512},
                    {"name": "title_company_location_skills_source", "max_len": 256},
                ],
            },
            "catboost": {"features_base": "dummy_path"},
        },
    }


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "title": ["Senior Python Developer", "ML Engineer"],
            "company": ["TechCo", "AI Labs"],
            "location": ["Moscow", "Saint Petersburg"],
            "source": ["hh", "getmatch"],
            "description": [
                "Требуется опытный разработчик с опытом от 3 до 5 лет",
                "Ищем специалиста с опытом более двух лет",
            ],
            "description_no_numbers": [
                "Требуется опытный разработчик с опытом от [NUMBER] до [NUMBER] лет",
                "Ищем специалиста с опытом более [NUMBER] лет",
            ],
            "skills": ["Python, SQL", "Python, ML"],
            "grade": ["senior", "middle"],
            "experience_from": [None, 2.0],
            "experience_to": [None, -1],
            "log_salary_from": [12.5, 12.0],
        }
    )


@pytest.fixture
def logger():
    return logging.getLogger("test")


@pytest.fixture
def feature_builder(logger, mock_config, sample_data):  # Add sample_data as dependency
    with patch("src.feature_building.main.load_config", return_value=mock_config):
        builder = FeatureBuilder(logger, is_inference=True)  # Always use inference mode for tests
        builder.data = sample_data  # Set data directly
        return builder


@pytest.fixture
def mock_config_transformer_disabled():
    config = {
        # Copy of mock_config with transformer disabled
        "is_test": True,
        "models": {
            "transformer": {
                "enabled": False,
            },
        },
        "features": {
            "preprocessed_data_base": "dummy_path",
            "output_base": "dummy_path",
            "target_base": "dummy_path",
            "target_name": "log_salary_from",
            "test_size": 1.0,
            "features": {
                "catboost": {
                    "text": ["title", "location", "company", "description_no_numbers_with_skills"],
                    "categorical": ["source"],
                    "numeric": ["experience_from", "experience_to_adjusted_10", "description_size"],
                },
                "transformer": {
                    "text": ["description_no_numbers", "title_company_location_skills_source"]
                },
                "bi_gru_cnn": {
                    "text": ["description_no_numbers", "title_company_location_skills_source"]
                },
            },
            "transformer": {
                "features_base": "dummy_path",
                "tokenizer": "intfloat/multilingual-e5-small",
                "tokenizer_test": "sergeyzh/rubert-tiny-turbo",
                "add_query_prefix": True,
                "feature_processing": [
                    {"name": "description_no_numbers", "max_len": 512},
                    {"name": "title_company_location_skills_source", "max_len": 256},
                ],
            },
            "catboost": {"features_base": "dummy_path"},
        },
    }
    return config


def test_data_property_setter(feature_builder):
    # Test valid DataFrame
    valid_df = pd.DataFrame({"col1": [1, 2]})
    feature_builder.data = valid_df
    assert feature_builder.data.equals(valid_df)

    # Test invalid input
    with pytest.raises(ValueError):
        feature_builder.data = [1, 2, 3]


def test_experience_extraction(feature_builder, sample_data):
    feature_builder.data = sample_data
    feature_builder.process_experience()

    # Check if experience was extracted correctly from first row
    assert feature_builder.data.loc[0, "experience_from"] == 3
    assert feature_builder.data.loc[0, "experience_to"] == 5

    # Check if existing experience values were preserved
    assert feature_builder.data.loc[1, "experience_from"] == 2
    assert feature_builder.data.loc[1, "experience_to"] == -1

    # Check if experience_to_adjusted_10 was calculated correctly
    assert feature_builder.data.loc[1, "experience_to_adjusted_10"] == 10


def test_description_size_feature(feature_builder, sample_data):
    feature_builder.data = sample_data
    feature_builder.merge_skills_and_descriptions()
    feature_builder.add_description_size_feature()

    # Check if description size is calculated correctly
    assert all(feature_builder.data["description_size"] > 0)
    assert isinstance(feature_builder.data["description_size"].iloc[0], np.int64)


def test_query_prefix_addition(feature_builder, sample_data):
    feature_builder.data = sample_data
    # First create the concatenated feature
    feature_builder.add_title_company_location_skills_source_feature()
    feature_builder.add_query_prefix_to_text_features()

    # Check if prefix was added to all text features
    for feature in feature_builder.text_features:
        assert all(feature_builder.data[feature].str.startswith("query: "))


def test_inference_mode(logger, mock_config, sample_data):
    with patch("src.feature_building.main.load_config", return_value=mock_config):
        builder = FeatureBuilder(logger, is_inference=True)
        builder.data = sample_data

        results = builder.build()  # Changed from builder.run()

        # Check if results contain expected features
        assert "transformer_features" in results
        assert "catboost_features" in results
        assert isinstance(results["catboost_features"], pd.DataFrame)
        assert not results["catboost_features"].empty

        # Check transformer features format
        for _, feature_data in results["transformer_features"].items():
            assert isinstance(feature_data["input_ids"], torch.Tensor)
            assert isinstance(feature_data["attention_mask"], torch.Tensor)
            assert isinstance(feature_data["token_type_ids"], torch.Tensor)
            assert set(feature_data.keys()) == {
                "input_ids",
                "attention_mask",
                "token_type_ids",
            }  # Check if only these keys are present


def test_error_handling(logger, mock_config):
    with patch("src.feature_building.main.load_config", return_value=mock_config):
        builder = FeatureBuilder(logger, is_inference=True)
        with pytest.raises(ValueError, match="No data available for processing"):
            builder.build()  # Changed from builder.run()


def test_text_concatenation(feature_builder, sample_data):
    feature_builder.data = sample_data
    feature_builder.add_title_company_location_skills_source_feature()

    result = feature_builder.data["title_company_location_skills_source"].iloc[0]

    # Check if all required fields are present in the concatenated text
    assert "Позиция:" in result
    assert "Компания:" in result
    assert "Место:" in result
    assert "Навыки:" in result
    assert "Источник:" in result

    # Check if actual values are present
    assert "Senior Python Developer" in result
    assert "TechCo" in result
    assert "Moscow" in result
    assert "Python, SQL" in result
    assert "hh" in result


def test_disabled_transformer(logger, mock_config_transformer_disabled, sample_data):
    with patch(
        "src.feature_building.main.load_config", return_value=mock_config_transformer_disabled
    ):
        builder = FeatureBuilder(logger, is_inference=True)
        builder.data = sample_data

        results = builder.build()

        # Check that transformer features are None when disabled
        assert results["transformer_features"] is None

        # Check that catboost features are still present and valid
        assert "catboost_features" in results
        assert isinstance(results["catboost_features"], pd.DataFrame)
        assert not results["catboost_features"].empty

        # Verify that text features don't have query prefix
        assert not any(sample_data["description_no_numbers"].str.contains("query:"))
