import pytest
import torch

from src.training.transformer.model import BaseTransformerModel
from src.training.transformer.model import SingleBERTWithCrossAttention
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
                "num_heads": 8,
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


def test_mlp_model_initialization(sample_config):
    """Test MLP model initialization."""
    model = SingleBERTWithMLP(sample_config)
    assert model is not None
    assert hasattr(model, "bert")
    assert hasattr(model, "mlp")
    assert isinstance(model, BaseTransformerModel)


def test_cross_attention_model_initialization(sample_config):
    """Test cross-attention model initialization."""
    model = SingleBERTWithCrossAttention(sample_config)
    assert model is not None
    assert hasattr(model, "bert")
    assert hasattr(model, "mlp")
    assert hasattr(model, "cross_attention")
    assert model.num_heads == sample_config["models"]["transformer"]["num_heads"]
    assert isinstance(model, BaseTransformerModel)


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


def test_models_with_variable_sequence_length(sample_config):
    """Test models with inputs of different sequence lengths."""
    batch_size = 2
    seq_length1 = 8
    seq_length2 = 12

    # Create inputs with different sequence lengths
    inputs = {
        "input1": torch.randint(0, 1000, (batch_size, seq_length1)),
        "attention_mask1": torch.ones(batch_size, seq_length1),
        "input2": torch.randint(0, 1000, (batch_size, seq_length2)),
        "attention_mask2": torch.ones(batch_size, seq_length2),
    }

    # Test MLP model
    mlp_model = SingleBERTWithMLP(sample_config)
    mlp_output = mlp_model(**inputs)
    assert mlp_output.shape == (batch_size, 1)

    # Test Cross-attention model
    cross_attention_model = SingleBERTWithCrossAttention(sample_config)
    cross_attn_output = cross_attention_model(**inputs)
    assert cross_attn_output.shape == (batch_size, 1)


def test_models_with_padding(sample_config):
    """Test models with padded inputs (some attention mask values are 0)."""
    batch_size = 2
    seq_length = 10

    # Create inputs with padding in the second half of each sequence
    inputs = {
        "input1": torch.randint(0, 1000, (batch_size, seq_length)),
        "attention_mask1": torch.tensor(
            [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]
        ),
        "input2": torch.randint(0, 1000, (batch_size, seq_length)),
        "attention_mask2": torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
        ),
    }

    # Test MLP model with padding
    mlp_model = SingleBERTWithMLP(sample_config)
    mlp_output = mlp_model(**inputs)
    assert mlp_output.shape == (batch_size, 1)

    # Test Cross-attention model with padding
    cross_attention_model = SingleBERTWithCrossAttention(sample_config)
    cross_attn_output = cross_attention_model(**inputs)
    assert cross_attn_output.shape == (batch_size, 1)


@pytest.mark.parametrize(
    "model_class",
    [SingleBERTWithMLP, SingleBERTWithCrossAttention],
    ids=["mlp_model", "cross_attention_model"],
)
def test_model_forward_cpu(sample_config, sample_inputs, model_class):
    """Test model forward pass on CPU."""
    model = model_class(sample_config)
    output = model(
        sample_inputs["input1"],
        sample_inputs["attention_mask1"],
        sample_inputs["input2"],
        sample_inputs["attention_mask2"],
    )

    # Common checks for all models
    assert output.shape == (2, 1)  # batch_size, 1
    assert output.dtype == torch.float32
    assert not torch.isnan(output).any(), "Output contains NaN values"
