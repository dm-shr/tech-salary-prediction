import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def split_tokenized_dict(tokenized_data, test_size=0.2, random_state=42):
    """Split tokenized data into train and test sets."""
    num_samples = len(next(iter(tokenized_data.values())))
    train_indices, test_indices = train_test_split(
        range(num_samples), test_size=test_size, random_state=random_state
    )
    train_data = {key: val[train_indices] for key, val in tokenized_data.items()}
    test_data = {key: val[test_indices] for key, val in tokenized_data.items()}
    return train_data, test_data


class DualTextDataset(Dataset):
    def __init__(self, tokenized_feature1, tokenized_feature2, targets):
        """Initialize dataset with pre-tokenized data."""
        self.tokenized_feature1 = tokenized_feature1
        self.tokenized_feature2 = tokenized_feature2
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        inputs1 = {key: val[idx] for key, val in self.tokenized_feature1.items()}
        inputs2 = {key: val[idx] for key, val in self.tokenized_feature2.items()}
        target = torch.tensor(self.targets[idx], dtype=torch.float)
        return inputs1, inputs2, target
