# Add these at the very top of app.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# import numpy as np
import streamlit as st

# from training.catboost.model import CatBoostModel
# from training.catboost.utils import create_pool_data
# from src.training.transformer.model import SingleBERTWithMLP
# from src.training.transformer.train import predict as transformer_predict
# from src.training.loader import load_models
from utils.utils import load_config, setup_logging

# from src.feature_building.main import FeatureBuilder


logger = setup_logging("streamlit_app")
config = load_config()
catboost_weight = config["models"]["blended"]["catboost_weight"]
transformer_weight = config["models"]["blended"]["transformer_weight"]


# # Load models once using cache
# @st.cache_resource
# def get_models():
#     try:
#         return load_models()
#     except Exception as e:
#         logger.error(f"Error loading models: {str(e)}")
#         st.error("Failed to load models. Please try again later.")
#         return {}

# Mock the app
@st.cache_resource
def healthcheck():
    logger.info("Starting the app...")
    return True


check = healthcheck()

# models = get_models()
# catboost_model = models.get("catboost")
# transformer_model = models.get("transformer")


# # Get feature builder once using cache
# @st.cache_resource
# def get_feature_builder():
#     return FeatureBuilder(setup_logging("streamlit_app"), is_inference=True)


# feature_builder = get_feature_builder()


# def predict_salary(
#     input_data, catboost_model: CatBoostModel, transformer_model: SingleBERTWithMLP = None
# ):
#     """Blend predictions from multiple models"""
#     # Predict using CatBoost model
#     # Extract features
#     catboost_features = input_data["catboost_features"]
#     # Turn features into CatBoost Pool object
#     catboost_pool = create_pool_data(config, catboost_features)
#     # Make prediction
#     catboost_pred = catboost_model.predict(catboost_pool)

#     # If transformer model is not supplied, return only catboost prediction
#     if transformer_model is None:
#         return catboost_pred

#     # Predict using transformer model
#     transformer_features = input_data["transformer_features"]
#     transformer_inputs1 = transformer_features["inputs1"]
#     transformer_inputs2 = transformer_features["inputs2"]
#     transformer_pred = transformer_predict(
#         transformer_model, transformer_inputs1, transformer_inputs2
#     )

#     # Calculate weighted average of predictions
#     blended_pred = catboost_weight * catboost_pred + transformer_weight * transformer_pred

#     return blended_pred


def main():
    st.title("Salary Prediction Service")

    # Input fields
    title = st.text_input("Job Title", value="Data Scientist")
    company = st.text_input("Company", value="Yandex")
    location = st.text_input("Location", value="Moscow")
    description = st.text_area("Job Description", value="We are looking for a Data Scientist")
    skills = st.text_area("Required Skills (comma-separated)", value="python,sql,ml")
    experience_from = st.number_input(
        "Minimum Years of Experience", min_value=0, max_value=20, value=0
    )
    experience_to = st.number_input(
        "Maximum Years of Experience", min_value=0, max_value=20, value=3
    )

    if st.button("Predict Salary"):
        try:
            # # Set input data to extract features from
            # feature_builder.prepare_set_inference_data(
            #     title,
            #     company,
            #     location,
            #     description,
            #     skills,
            #     experience_from,
            #     experience_to,
            # )

            # # Generate features
            # features = feature_builder.build()

            # # Make prediction
            # predicted_salary = predict_salary(features, catboost_model, transformer_model)

            # # Reverse log transformation and scale
            # predicted_salary = np.exp(predicted_salary) * 1000

            # st.success(f"Predicted Salary: RUR {predicted_salary[0]:,.2f}")
            logger.info("healthcheck: %s", check)
            # log the input data
            logger.info(
                "title: %s, company: %s, location: %s, description: %s, skills: %s, experience_from: %s, experience_to: %s",
                title,
                company,
                location,
                description,
                skills,
                experience_from,
                experience_to,
            )
            st.success("Predicted Salary: RUR ???")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")


if __name__ == "__main__":
    main()
