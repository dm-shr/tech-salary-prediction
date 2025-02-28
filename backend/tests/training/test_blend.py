import numpy as np
import pytest

from src.training.blend import blend_and_evaluate
from src.training.blend import calculate_metrics
from src.training.blend import mean_confidence_interval


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing."""
    n_seeds = 3
    n_epochs = 4
    n_samples = 5

    # Create dummy predictions
    catboost_preds = np.random.rand(n_seeds, n_samples)
    transformer_preds = np.random.rand(n_seeds, n_epochs, n_samples)
    y_true = np.random.rand(n_seeds, n_samples)

    return catboost_preds, transformer_preds, y_true


def test_mean_confidence_interval():
    """Test confidence interval calculation."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, lower, upper = mean_confidence_interval(data, alpha=0.05)

    assert lower < mean < upper
    assert mean == pytest.approx(3.0)


def test_calculate_metrics():
    """Test basic metrics calculation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1, 10.0])

    r2, mae, rmse = calculate_metrics(y_true, y_pred)

    assert r2 <= 1
    assert mae > 0
    assert rmse > 0
    assert rmse > mae


def test_r2_calculation():
    """Test R^2 calculation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.mean(y_true) * np.ones_like(y_true)  # Predict the mean

    r2, _, _ = calculate_metrics(y_true, y_pred)

    assert r2 == pytest.approx(0.0, abs=1e-6)


def test_blend_and_evaluate(sample_predictions):
    """Test the complete blending and evaluation pipeline."""
    catboost_preds, transformer_preds, y_true = sample_predictions

    result, _ = blend_and_evaluate(
        catboost_preds, transformer_preds, y_true, blend_weights=(0.5, 0.5)
    )

    assert "best_epoch" in result
    assert "metrics" in result
    assert "r2" in result["metrics"]
    assert len(result["metrics"]["r2"]) == 3  # mean, lower, upper
