# from typing import Dict
from fastapi import FastAPI
from pydantic import BaseModel
from src.utils.utils import load_config
from src.utils.utils import setup_logging

# from fastapi import HTTPException

# from src.feature_building.main import FeatureBuilder
# from training.catboost.model import CatBoostModel
# from training.transformer.model import SingleBERTWithMLP

logger = setup_logging("fastapi")
config = load_config()

# Load models (replace with your actual model loading logic)
# def load_models():
#     logger.info("Loading models...")
#     try:
#         catboost_model: CatBoostModel
#         transformer_model: SingleBERTWithMLP
#         catboost_model, transformer_model = load_models()
#         logger.info("Models loaded successfully")
#         return {"catboost": catboost_model, "transformer": transformer_model}
#     except Exception as e:
#         logger.error(f"Error loading models: {e}")
#         raise HTTPException(status_code=500, detail="Failed to load models")

# models = load_models()
# feature_builder = FeatureBuilder(setup_logging("fastapi"), is_inference=True)

app = FastAPI()


class InferenceInput(BaseModel):
    title: str
    company: str
    location: str
    description: str
    skills: str
    experience_from: int
    experience_to: int


@app.get("/healthcheck")
async def healthcheck():
    logger.info("Health check requested")
    return {"status": "ok"}


# @app.post("/predict")
# async def predict(input_data: InferenceInput):
#     try:
#         # Prepare inference data
#         feature_builder.prepare_set_inference_data(
#             input_data.title,
#             input_data.company,
#             input_data.location,
#             input_data.description,
#             input_data.skills,
#             input_data.experience_from,
#             input_data.experience_to,
#         )

#         # Build features
#         features = feature_builder.build()

#         # Make prediction
#         predicted_salary = predict_salary(
#             features, models["catboost"], models["transformer"]
#         )

#         # Reverse log transformation and scale
#         predicted_salary = np.exp(predicted_salary) * 1000

#         logger.info(f"Predicted salary: {predicted_salary[0]}")
#         return {"predicted_salary": predicted_salary[0]}

#     except Exception as e:
#         logger.error(f"Error during prediction: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
@app.post("/predict")
async def predict(input_data: InferenceInput):
    logger.info("healthcheck: %s", "ok")
    return {"predicted_salary": "???"}
