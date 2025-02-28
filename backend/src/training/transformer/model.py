from abc import ABC
from abc import abstractmethod

import torch
import torch.nn as nn
from transformers import AutoModel


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # Apply multi-head attention
        attn_output, _ = self.attention(
            query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )

        # Add residual connection and layer norm
        output = self.layer_norm(query + attn_output)  # residual connection with query
        return output


class BaseTransformerModel(nn.Module, ABC):
    """Abstract base class for transformer models that process two text inputs."""

    def __init__(self, config):
        # fmt: off
        super().__init__()
        # fmt: on
        # Common configuration for all transformer models
        model_name = (
            config["models"]["transformer"]["model_name"]
            if not config["is_test"]
            else config["models"]["transformer"]["model_name_test"]
        )
        self.hidden_size = (
            config["models"]["transformer"]["hidden_size"]
            if not config["is_test"]
            else config["models"]["transformer"]["hidden_size_test"]
        )
        self.mlp_hidden_size = config["models"]["transformer"]["mlp_hidden_size"]

        # Initialize BERT model
        self.bert = AutoModel.from_pretrained(model_name)

        # Define MLP head (common for both models)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, 1),  # Regression output
        )

    @abstractmethod
    def forward(
        self,
        input1: torch.Tensor,
        attention_mask1: torch.Tensor,
        input2: torch.Tensor,
        attention_mask2: torch.Tensor,
    ) -> torch.Tensor:
        """Process two input sequences through the model."""
        pass


class SingleBERTWithCrossAttention(BaseTransformerModel):
    """Single BERT model with cross-attention between the two text features.
    Here, query is the first text feature and key, value are the second text feature."""

    # fmt: off
    def __init__(self, config):
        super().__init__(config)
        self.num_heads = config["models"]["transformer"]["num_heads"]
        self.cross_attention = CrossAttentionLayer(self.hidden_size, num_heads=self.num_heads)
    # fmt: on

    def forward(
        self,
        input1: torch.Tensor,
        attention_mask1: torch.Tensor,
        input2: torch.Tensor,
        attention_mask2: torch.Tensor,
    ):
        # Get BERT outputs
        outputs1 = self.bert(input_ids=input1, attention_mask=attention_mask1).last_hidden_state
        outputs2 = self.bert(input_ids=input2, attention_mask=attention_mask2).last_hidden_state

        # Get raw averaged embeddings for the feature 2
        averaged_pool2 = average_pool(outputs2, attention_mask2)

        # prepare key_padding_mask
        key_padding_mask = (
            attention_mask2 == 0
        ).bool()  # True indicates positions to exclude (padding tokens). Shape: (batch_size, source_len)

        # prepare attention mask
        # Step 1: Expand to match query and key dimensions
        # original shape: (batch_size, target_len)
        attn_mask = attention_mask1.unsqueeze(
            2
        )  # create a new dimension at the end to be able to expand. Shape: (batch_size, target_len, 1)
        attn_mask = attn_mask.expand(
            -1, -1, attention_mask2.size(1)
        )  # expand to match the source_len. Shape: (batch_size, target_len, source_len)

        # Step 2: Adjust for multi-head attention
        attn_mask = attn_mask.unsqueeze(
            1
        )  # Add head dimension at position 1. Shape: (batch_size, 1, target_len, source_len)
        attn_mask = attn_mask.repeat(
            1, self.num_heads, 1, 1
        )  # Repeat for each head. Shape: (batch_size, num_heads, target_len, source_len)
        attn_mask = attn_mask.view(
            -1, attention_mask1.size(1), attention_mask2.size(1)
        )  # Merge batch and head dimensions. Shape: (batch_size * num_heads, target_len, source_len)

        # Step 3: Convert to boolean mask
        attn_mask = (
            attn_mask == 0
        ).bool()  # True indicates positions to exclude (padding tokens). Shape: (batch_size * num_heads, target_len, source_len)

        # Apply cross-attention
        # NOTE: attn_mask is not used here!
        attended_features = self.cross_attention(
            query=outputs1,
            key=outputs2,
            value=outputs2,
            key_padding_mask=key_padding_mask,
            # attn_mask=attn_mask,
        )

        # Get averaged embedding from attended features
        averaged_pool1 = average_pool(attended_features, attention_mask1)

        # Concatenate the two averaged embeddings
        combined = torch.cat(
            [averaged_pool1, averaged_pool2], dim=1
        )  # Shape: (batch_size, 2 * hidden_size)

        # Pass through MLP head
        output = self.mlp(combined)

        return output


class SingleBERTWithMLP(BaseTransformerModel):
    """Basic single BERT model with an MLP head to ingest the concatenated CLS embeddings."""

    def forward(self, input1, attention_mask1, input2, attention_mask2):
        # Pass both inputs through the same BERT model
        cls1 = self.bert(input_ids=input1, attention_mask=attention_mask1).last_hidden_state[
            :, 0, :
        ]  # CLS token for input1
        cls2 = self.bert(input_ids=input2, attention_mask=attention_mask2).last_hidden_state[
            :, 0, :
        ]  # CLS token for input2

        # Concatenate CLS embeddings
        combined_cls = torch.cat([cls1, cls2], dim=-1)  # Shape: [batch_size, 2 * hidden_size]

        # Pass through MLP head
        output = self.mlp(combined_cls)
        return output
