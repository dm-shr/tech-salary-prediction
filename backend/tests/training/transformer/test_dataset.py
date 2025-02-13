import pytest
import torch

from src.training.transformer.dataset import DualTextDataset
from src.training.transformer.dataset import split_tokenized_dict


@pytest.fixture
def sample_tokenized_data():
    """Create sample tokenized data for testing."""
    return {
        "input_ids": torch.randint(0, 1000, (10, 5)),  # 10 samples, length 5
        "attention_mask": torch.ones(10, 5),
    }


@pytest.fixture
def sample_dataset_inputs():
    """Create sample inputs for dataset testing."""
    tokenized_feature1 = {
        "input_ids": torch.randint(0, 1000, (10, 5)),
        "attention_mask": torch.ones(10, 5),
    }
    tokenized_feature2 = {
        "input_ids": torch.randint(0, 1000, (10, 5)),
        "attention_mask": torch.ones(10, 5),
    }
    targets = torch.randn(10)

    return tokenized_feature1, tokenized_feature2, targets


def test_split_tokenized_dict(sample_tokenized_data):
    """Test tokenized data splitting."""
    train_data, test_data = split_tokenized_dict(
        sample_tokenized_data, test_size=0.2, random_state=42
    )

    assert len(train_data["input_ids"]) == 8  # 80% of 10
    assert len(test_data["input_ids"]) == 2  # 20% of 10
    assert "attention_mask" in train_data
    assert "attention_mask" in test_data


def test_dual_text_dataset(sample_dataset_inputs):
    """Test DualTextDataset functionality."""
    tokenized_feature1, tokenized_feature2, targets = sample_dataset_inputs
    dataset = DualTextDataset(tokenized_feature1, tokenized_feature2, targets)

    assert len(dataset) == 10

    # Test getting an item
    inputs1, inputs2, target = dataset[0]
    assert isinstance(inputs1, dict)
    assert isinstance(inputs2, dict)
    assert isinstance(target, torch.Tensor)
    assert target.dim() == 0  # scalar
