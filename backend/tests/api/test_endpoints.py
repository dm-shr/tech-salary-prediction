from unittest.mock import patch

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi import status

from src.feature_building.main import TranslationResult
from src.training.catboost.model import CatBoostModel
from src.training.transformer.model import SingleBERTWithMLP


API_KEY = "test_api_key"  # Define a test API key
API_KEYS = f"{API_KEY},another_key"  # Define a test API keys


def test_healthcheck_transformer_enabled(test_client, mock_config_transformer_enabled):
    """Test healthcheck endpoint when transformer is enabled."""
    with patch("src.fastapi_app.config", mock_config_transformer_enabled):
        response = test_client.get("/healthcheck")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "ok", "transformer_enabled": True}


def test_healthcheck_transformer_disabled(test_client, mock_config_transformer_disabled):
    """Test healthcheck endpoint when transformer is disabled."""
    with patch("src.fastapi_app.config", mock_config_transformer_disabled):
        response = test_client.get("/healthcheck")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "ok", "transformer_enabled": False}


def test_predict_with_transformer(
    test_client,
    mock_config_transformer_enabled,
    mock_catboost_model: CatBoostModel,
    mock_transformer_model: SingleBERTWithMLP,
    sample_input,
):
    """Test prediction endpoint with both models enabled."""
    with patch("src.fastapi_app.config", mock_config_transformer_enabled), patch(
        "src.feature_building.main.FeatureBuilder._translate_with_gemini"
    ) as mock_translate, patch("src.fastapi_app.predict_salary") as mock_predict_salary, patch(
        "src.fastapi_app.FeatureBuilder.build"
    ) as mock_feature_builder, patch(
        "src.fastapi_app.catboost_model", new=mock_catboost_model
    ), patch(
        "src.fastapi_app.transformer_model", new=mock_transformer_model
    ), patch(
        "src.fastapi_app.API_KEYS", [API_KEY]
    ):  # Patch API_KEYS directly
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

        response = test_client.post("/predict", json=sample_input, headers={"X-API-Key": API_KEY})

        assert response.status_code == status.HTTP_200_OK
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
    """Test prediction endpoint with only CatBoost model."""
    with patch("src.fastapi_app.config", mock_config_transformer_disabled), patch(
        "src.feature_building.main.FeatureBuilder._translate_with_gemini"
    ) as mock_translate, patch("src.fastapi_app.predict_salary") as mock_predict_salary, patch(
        "src.fastapi_app.FeatureBuilder.build"
    ) as _, patch(
        "src.fastapi_app.catboost_model", new=mock_catboost_model
    ), patch(
        "src.fastapi_app.transformer_model", new=None
    ), patch(
        "src.fastapi_app.API_KEYS", [API_KEY]
    ):  # Patch API_KEYS directly
        # Mock the translation to return the original values
        mock_translate.return_value = TranslationResult(
            company=sample_input["company"],
            description=sample_input["description"],
            location=sample_input["location"],
        )

        # Mock the predict_salary function
        mock_predict_salary.return_value = 10.0  # Dummy prediction

        response = test_client.post("/predict", json=sample_input, headers={"X-API-Key": API_KEY})

        assert response.status_code == status.HTTP_200_OK
        assert "predicted_salary" in response.json()
        assert isinstance(response.json()["predicted_salary"], float)
        assert response.json()["predicted_salary"] > 0
        assert response.json()["predicted_salary"] == pytest.approx(np.exp(10.0) * 1000)


def test_predict_invalid_input(test_client):
    """Test prediction endpoint with missing required fields."""
    invalid_input = {
        "title": "Data Scientist",
        # missing required fields
    }

    try:
        test_client.post("/predict", json=invalid_input, headers={"X-API-Key": API_KEY})
    except HTTPException as e:
        assert e.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


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
    """Test prediction endpoint with invalid field values."""
    modified_input = sample_input.copy()
    modified_input[field] = invalid_value

    try:
        test_client.post("/predict", json=modified_input, headers={"X-API-Key": API_KEY})
    except HTTPException as e:
        assert e.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_predict_model_error(
    test_client, mock_config_transformer_enabled, mock_catboost_model, sample_input
):
    """Test prediction endpoint when model fails."""
    with patch("src.fastapi_app.config", mock_config_transformer_enabled), patch(
        "src.fastapi_app.FeatureBuilder.build"
    ) as mock_feature_builder:
        mock_feature_builder.side_effect = Exception("Model error")
    try:
        test_client.post("/predict", json=sample_input, headers={"X-API-Key": API_KEY})
    except HTTPException as e:
        assert e.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert e.detail == "Model error"


def test_predict_missing_api_key(test_client, sample_input):
    """Test prediction endpoint with missing API key."""
    try:
        test_client.post("/predict", json=sample_input)
    except HTTPException as e:
        assert e.status_code == status.HTTP_401_UNAUTHORIZED
        assert e.detail == "Invalid API Key"


def test_predict_invalid_api_key(test_client, sample_input):
    """Test prediction endpoint with invalid API key."""
    with patch.dict("os.environ", {"API_KEYS": API_KEYS}):
        response = test_client.post(
            "/predict", json=sample_input, headers={"X-API-Key": "invalid_key"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert response.json() == {"detail": "Invalid API Key"}
