import asyncio
import os
import sys
from typing import Optional

import numpy as np
import streamlit as st

from src.feature_building.main import FeatureBuilder
from src.training.catboost.model import CatBoostModel
from src.training.catboost.utils import create_pool_data
from src.training.loader import load_models
from src.utils.utils import load_config
from src.utils.utils import setup_logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import transformer dependencies conditionally
config = load_config()
if config["models"]["transformer"]["enabled"]:
    from src.training.transformer.model import SingleBERTWithMLP
    from src.training.transformer.train import predict as transformer_predict

logger = setup_logging("streamlit_app")
catboost_weight = config["models"]["blended"]["catboost_weight"]
transformer_weight = config["models"]["blended"]["transformer_weight"]


# Load models once using cache
@st.cache_resource
def get_models():
    """Load models based on configuration."""
    try:
        models = load_models()
        if not models.get("catboost"):
            raise ValueError("CatBoost model failed to load")

        if config["models"]["transformer"]["enabled"]:
            if not models.get("transformer"):
                raise ValueError("Transformer enabled but failed to load")
            st.success("Both models loaded successfully")
        else:
            st.info("Running with CatBoost only (Transformer disabled)")

        return models
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        st.error("Failed to load models. Please try again later.")
        return {}


# Get feature builder once using cache
@st.cache_resource
def get_feature_builder():
    return FeatureBuilder(setup_logging("streamlit_app"), is_inference=True)


def predict_salary(
    input_data: dict,
    catboost_model: CatBoostModel,
    transformer_model: Optional["SingleBERTWithMLP"] = None,
) -> float:
    """Make predictions using available models."""
    try:
        # Get CatBoost prediction
        catboost_features = input_data["catboost_features"]
        catboost_pool = create_pool_data(config, catboost_features)
        catboost_pred = catboost_model.predict(catboost_pool)

        # Return only CatBoost prediction if transformer is disabled
        if not config["models"]["transformer"]["enabled"]:
            logger.info("Using CatBoost prediction only")
            return catboost_pred

        # Get transformer prediction and blend
        transformer_features = input_data["transformer_features"]
        transformer_pred = transformer_predict(
            transformer_model,
            transformer_features["description_no_numbers"],
            transformer_features["title_company_location_skills_source"],
        )

        # Blend predictions
        blended_pred = catboost_weight * catboost_pred + transformer_weight * transformer_pred
        logger.info("Using blended prediction")
        return blended_pred

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise


# Initialize models and feature builder
models = get_models()
feature_builder = get_feature_builder()
catboost_model = models.get("catboost")
transformer_model = (
    models.get("transformer") if config["models"]["transformer"]["enabled"] else None
)


async def main():
    st.title("Salary Prediction Service")

    # Show model configuration
    if config["models"]["transformer"]["enabled"]:
        st.info("Running with blended model (CatBoost + Transformer)")
    else:
        st.info("Running with CatBoost model only")

    # Input fields
    title = st.text_input("Job Title", value="Machine Learning Engineer")
    company = st.text_input("Company", value="Spotify")
    location = st.text_input("Location", value="Stockholm")
    description = st.text_area(
        "Job Description",
        value="We are seeking a Machine Learning Engineer to join our team. "
        "The ideal candidate will have experience in developing and deploying ML models, "
        "working with large datasets, and implementing end-to-end ML pipelines. "
        "Key responsibilities include model development, experimentation, "
        "and collaboration with cross-functional teams. "
        "Strong programming skills in Python and experience with deep learning frameworks required.",
    )
    skills = st.text_area("Required Skills (comma-separated)", value="Python, SQL, PyTorch")
    experience_from = st.number_input(
        "Minimum Years of Experience", min_value=0, max_value=20, value=3
    )
    experience_to = st.number_input(
        "Maximum Years of Experience", min_value=0, max_value=20, value=6
    )

    if st.button("Predict Salary"):
        try:
            # Set input data
            await feature_builder.prepare_set_inference_data(
                title, company, location, description, skills, experience_from, experience_to
            )

            # Generate features
            features = feature_builder.build()

            # Make prediction
            predicted_salary = predict_salary(features, catboost_model, transformer_model)

            # Reverse log transformation and scale
            final_salary = np.exp(predicted_salary) * 1000

            # Currency conversion and formatting
            CURRENCY_CONVERSION = 0.43
            predicted_salary_value = float(final_salary) * CURRENCY_CONVERSION
            rounded_salary = round(predicted_salary_value / 100) * 100  # Round to nearest hundred
            formatted_salary = str(int(rounded_salary)).replace(
                r"\B(?=(\d{3})+(?!\d))", " "
            )  # Add space for thousands separator

            st.success(f"Predicted Salary: {formatted_salary} SEK/month")
            logger.info(f"Prediction successful: {formatted_salary} SEK/month")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            logger.error(f"Prediction error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
