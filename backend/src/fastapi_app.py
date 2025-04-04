import os
import time
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import Depends
from fastapi import FastAPI
from fastapi import Header
from fastapi import HTTPException
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client import Gauge
from prometheus_client import generate_latest
from prometheus_client import Histogram
from pydantic import BaseModel
from supabase import create_client

from src.feature_building.main import FeatureBuilder
from src.training.catboost.model import CatBoostModel
from src.training.catboost.utils import create_pool_data
from src.training.loader import load_models
from src.utils.utils import load_config
from src.utils.utils import setup_logging

config = load_config()

# Import transformer dependencies conditionally
if config["models"]["transformer"]["enabled"]:
    from src.training.transformer.model import SingleBERTWithMLP
    from src.training.transformer.train import predict as transformer_predict


logger = setup_logging("fastapi")
catboost_weight = config["models"]["blended"]["catboost_weight"]
transformer_weight = config["models"]["blended"]["transformer_weight"]

load_dotenv()

# Prometheus metrics
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Latency of salary predictions")
PREDICTED_SALARY = Gauge("predicted_salary", "Value of predicted salary")


def get_models():
    """Load models based on configuration."""
    try:
        models = load_models()
        if not models.get("catboost"):
            logger.error(
                "CatBoost model failed to load.  Please check the model path and S3 access."
            )
            raise ValueError("CatBoost model failed to load")

        if config["models"]["transformer"]["enabled"]:
            if not models.get("transformer"):
                logger.error(
                    "Transformer model failed to load. Please check the model path and S3 access."
                )
                raise ValueError("Transformer enabled but failed to load")
            logger.info("Both models loaded successfully")
        else:
            logger.info("CatBoost model loaded successfully (Transformer disabled)")

        return models
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


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
        if transformer_features is not None:
            transformer_pred = transformer_predict(
                transformer_model,
                transformer_features["description_no_numbers"],
                transformer_features["title_company_location_skills_source"],
            )

            # Blend predictions
            blended_pred = catboost_weight * catboost_pred + transformer_weight * transformer_pred
            logger.info("Using blended prediction")
            return blended_pred
        else:
            logger.info("Transformer features are None, using CatBoost prediction only")
            return catboost_pred

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise


# Initialize models and feature builder
models = get_models()
catboost_model = models["catboost"]
transformer_model = (
    models.get("transformer") if config["models"]["transformer"]["enabled"] else None
)
feature_builder = FeatureBuilder(setup_logging("fastapi"), is_inference=True)

# FastAPI setup
app = FastAPI()

# Configure CORS
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
logger.info("Allowed origins: %s", origins)

# API Key Authentication
API_KEYS = os.getenv("API_KEYS", "").split(",")


async def verify_api_key(x_api_key: str = Header(...)):
    valid_keys = [key.strip() for key in API_KEYS if key.strip()]
    if x_api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["POST"],  # Only allow needed methods
    allow_headers=["X-API-Key", "Content-Type"],  # Only allow needed headers
    expose_headers=[],
)

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.warning("Supabase URL and Key not set, logging to Supabase will be disabled")
    supabase = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def store_prediction(user_id: str, chat_history: dict):
    """Stores prediction data in Supabase."""
    if not supabase:
        logger.warning("Supabase client not initialized, skipping logging")
        return None

    data = {
        "user_id": user_id,
        "chat_history": chat_history,
        "app_type": "tech-salary-prediction",
    }

    try:
        response = supabase.table("conversations").insert(data).execute()
        logger.info(f"Prediction stored in Supabase for user: {user_id}")
        return response
    except Exception as e:
        logger.error(f"Error storing prediction in Supabase: {e}")
        return None


class InferenceInput(BaseModel):
    userId: str
    title: str
    company: str
    location: str
    description: str
    skills: str
    experience_from: int
    experience_to: int


@app.get("/healthcheck")
async def healthcheck():
    transformer_status = "enabled" if config["models"]["transformer"]["enabled"] else "disabled"
    logger.info(f"Health check requested (Transformer: {transformer_status})")
    return {"status": "ok", "transformer_enabled": config["models"]["transformer"]["enabled"]}


@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(input_data: InferenceInput):
    start_time = time.time()
    try:
        # Prepare inference data
        await feature_builder.prepare_set_inference_data(
            input_data.title,
            input_data.company,
            input_data.location,
            input_data.description,
            input_data.skills,
            input_data.experience_from,
            input_data.experience_to,
        )

        # Build features
        features = feature_builder.build()

        # Make prediction
        predicted_salary = predict_salary(features, catboost_model, transformer_model)

        # Reverse log transformation and scale
        final_salary = float(np.exp(predicted_salary) * 1000)

        # Track the predicted salary value
        PREDICTED_SALARY.set(final_salary)

        # Create response
        response = {"predicted_salary": final_salary}

        # Log to Supabase
        input_dict = input_data.dict()
        user_id = input_dict.pop("userId")  # Extract userId and remove from input_dict
        chat_history = input_dict
        chat_history["predicted_salary"] = final_salary

        # Store prediction in Supabase
        store_prediction(user_id, chat_history)

        logger.info(f"Predicted salary: {final_salary:,.2f}")
        return response

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
