import pytest
import torch

from src.training.transformer.model import SingleBERTWithMLP


@pytest.fixture
def sample_config():
    """Create a sample config for model testing."""
    return {
        "is_test": True,
        "models": {
            "transformer": {
                "model_name_test": "bert-base-uncased",
                "hidden_size_test": 768,
                "mlp_hidden_size": 256,
            }
        },
    }


@pytest.fixture
def sample_inputs():
    """Create sample inputs for model testing."""
    batch_size = 2
    seq_length = 10
    return {
        "input1": torch.randint(0, 1000, (batch_size, seq_length)),
        "attention_mask1": torch.ones(batch_size, seq_length),
        "input2": torch.randint(0, 1000, (batch_size, seq_length)),
        "attention_mask2": torch.ones(batch_size, seq_length),
    }


def test_model_initialization(sample_config):
    """Test model initialization."""
    model = SingleBERTWithMLP(sample_config)
    assert model is not None
    assert hasattr(model, "bert")
    assert hasattr(model, "mlp")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_forward(sample_config, sample_inputs):
    """Test model forward pass."""
    model = SingleBERTWithMLP(sample_config)
    model.to("cuda")

    # Move inputs to GPU
    inputs = {k: v.to("cuda") for k, v in sample_inputs.items()}

    output = model(
        inputs["input1"], inputs["attention_mask1"], inputs["input2"], inputs["attention_mask2"]
    )

    assert output.shape == (2, 1)  # batch_size, 1
