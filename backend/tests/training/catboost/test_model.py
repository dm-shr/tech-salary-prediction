import numpy as np
import pandas as pd
import pytest
from catboost import Pool
from src.training.catboost.model import CatBoostModel
from src.training.catboost.utils import create_pool_data


@pytest.fixture
def sample_config():
    """Create a sample config for model testing."""
    return {
        "is_test": True,
        "models": {
            "catboost": {
                "learning_rate": "0.05",
                "bagging_temperature": "0",
                "random_strength": "10",
                "l2_leaf_reg": "0",
                "depth": "4",
                "iterations": "2000",
                "iterations_test": "100",
                "early_stopping_rounds": "100",
                "loss_function": "RMSE",
                "eval_metric": "R2",
                "tokenizer_id": "Space",
                "dictionary_id": "Word",
                "max_dictionary_size": "50000",
                "occurrence_lower_bound": "25",
                "occurrence_lower_bound_test": "0",
                "gram_order": "1",
                "save_dir": "test_models/catboost",
            }
        },
        "features": {
            "features": {
                "catboost": {
                    "text": ["title", "description"],
                    "categorical": ["source"],
                    "numeric": ["experience_from"],
                }
            }
        },
    }


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    n_samples = 100

    data = {
        "title": [f"Job Title {i}" for i in range(n_samples)],
        "description": [f"Description {i}" for i in range(n_samples)],
        "source": ["source_A" if i % 2 == 0 else "source_B" for i in range(n_samples)],
        "experience_from": np.random.uniform(0, 10, n_samples),
    }

    X = pd.DataFrame(data)
    y = np.random.normal(5, 1, n_samples)  # Target values

    return X, y


def test_model_initialization(sample_config):
    """Test model initialization with config."""
    model = CatBoostModel.model_from_config(sample_config, seed=42)
    assert model is not None
    assert model.model is not None

    params = model.get_params()
    assert params["iterations"] == 100  # test mode
    assert params["learning_rate"] == 0.05
    assert params["depth"] == 4


def test_create_pool_data(sample_config, sample_data):
    """Test creation of CatBoost Pool object."""
    X, y = sample_data
    pool = create_pool_data(sample_config, X, y)

    assert isinstance(pool, Pool)
    assert pool.num_row() == len(X)
    assert len(pool.get_feature_names()) == len(X.columns)


def test_model_training(sample_config, sample_data):
    """Test model training and evaluation."""
    X, y = sample_data
    model = CatBoostModel.model_from_config(sample_config, seed=42)

    # Create pool
    train_pool = create_pool_data(sample_config, X, y)

    # Train without evaluation set
    metrics = model.train_and_evaluate_with_metrics(
        train_pool=train_pool, X_train=X, y_train=y, to_eval=False
    )

    assert "r2" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert all(isinstance(v, float) for v in metrics.values())
    assert all(v >= 0 for v in metrics.values())


def test_model_prediction(sample_config, sample_data):
    """Test model prediction functionality."""
    X, y = sample_data
    model = CatBoostModel.model_from_config(sample_config, seed=42)

    # Train the model
    train_pool = create_pool_data(sample_config, X, y)
    model.model.fit(train_pool)

    # Make predictions
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert isinstance(predictions, np.ndarray)
    assert not np.any(np.isnan(predictions))
