from unittest.mock import patch

import numpy as np
import pytest

from src.feature_building.main import TranslationResult
from src.training.catboost.model import CatBoostModel
from src.training.transformer.model import SingleBERTWithMLP


def test_healthcheck_transformer_enabled(test_client, mock_config_transformer_enabled):
    with patch("src.api.config", mock_config_transformer_enabled):
        response = test_client.get("/healthcheck")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "transformer_enabled": True}


def test_healthcheck_transformer_disabled(test_client, mock_config_transformer_disabled):
    with patch("src.api.config", mock_config_transformer_disabled):
        response = test_client.get("/healthcheck")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "transformer_enabled": False}


def test_predict_with_transformer(
    test_client,
    mock_config_transformer_enabled,
    mock_catboost_model: CatBoostModel,
    mock_transformer_model: SingleBERTWithMLP,
    sample_input,
):
    with patch("src.api.config", mock_config_transformer_enabled), patch(
        "src.feature_building.main.FeatureBuilder._translate_with_gemini"
    ) as mock_translate, patch("src.api.predict_salary") as mock_predict_salary, patch(
        "src.api.FeatureBuilder.build"
    ) as mock_feature_builder, patch(
        "src.api.catboost_model", new=mock_catboost_model
    ), patch(
        "src.api.transformer_model", new=mock_transformer_model
    ):

        # Mock the translation to return the original values
        mock_translate.return_value = TranslationResult(
            company=sample_input["company"],
            description=sample_input["description"],
            location=sample_input["location"],
        )

        # Define mock features
        mock_features = {
            "catboost_features": [1, 2, 3],  # Dummy catboost features
            "transformer_features": {
                "description_no_numbers": [[1, 2, 3]],  # Dummy transformer features
                "title_company_location_skills_source": [[4, 5, 6]],  # Dummy transformer features
            },
        }
        mock_feature_builder.return_value = mock_features

        # Mock the predict_salary function
        mock_predict_salary.return_value = 10.0  # Dummy prediction

        response = test_client.post("/predict", json=sample_input)

        assert response.status_code == 200
        assert "predicted_salary" in response.json()
        assert isinstance(response.json()["predicted_salary"], float)
        assert response.json()["predicted_salary"] > 0
        assert response.json()["predicted_salary"] == pytest.approx(np.exp(10.0) * 1000)

        # Assert that predict_salary is called with the correct arguments
        mock_predict_salary.assert_called_once_with(
            mock_features, mock_catboost_model, mock_transformer_model
        )


def test_predict_without_transformer(
    test_client, mock_config_transformer_disabled, mock_catboost_model: CatBoostModel, sample_input
):
    with patch("src.api.config", mock_config_transformer_disabled), patch(
        "src.feature_building.main.FeatureBuilder._translate_with_gemini"
    ) as mock_translate, patch("src.api.predict_salary") as mock_predict_salary, patch(
        "src.api.FeatureBuilder.build"
    ) as _, patch(
        "src.api.catboost_model", new=mock_catboost_model
    ), patch(
        "src.api.transformer_model", new=None
    ):
        # Mock the translation to return the original values
        mock_translate.return_value = TranslationResult(
            company=sample_input["company"],
            description=sample_input["description"],
            location=sample_input["location"],
        )

        # Mock the predict_salary function
        mock_predict_salary.return_value = 10.0  # Dummy prediction

        response = test_client.post("/predict", json=sample_input)

        assert response.status_code == 200
        assert "predicted_salary" in response.json()
        assert isinstance(response.json()["predicted_salary"], float)
        assert response.json()["predicted_salary"] > 0
        assert response.json()["predicted_salary"] == pytest.approx(np.exp(10.0) * 1000)


def test_predict_invalid_input(test_client):
    invalid_input = {
        "title": "Data Scientist",
        # missing required fields
    }

    response = test_client.post("/predict", json=invalid_input)
    assert response.status_code == 422  # Validation error


@pytest.mark.parametrize(
    "field,invalid_value",
    [
        ("experience_from", -1),
        ("experience_to", 100),
        ("title", ""),
        ("skills", None),
    ],
)
def test_predict_invalid_field_values(test_client, sample_input, field, invalid_value):
    modified_input = sample_input.copy()
    modified_input[field] = invalid_value

    with patch("src.api.feature_builder.prepare_set_inference_data") as mock_prepare:
        mock_prepare.side_effect = ValueError("Invalid input")  # Prevent feature building
        response = test_client.post("/predict", json=modified_input)
        assert response.status_code == 422


def test_predict_model_error(
    test_client, mock_config_transformer_enabled, mock_catboost_model, sample_input
):
    with patch("src.api.config", mock_config_transformer_enabled), patch(
        "src.api.FeatureBuilder.build"
    ) as mock_feature_builder:
        mock_feature_builder.side_effect = Exception("Model error")

        response = test_client.post("/predict", json=sample_input)
        assert response.status_code == 500
        assert "Model error" in response.json()["detail"]
