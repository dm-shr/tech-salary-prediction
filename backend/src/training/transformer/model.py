# model
import torch
import torch.nn as nn
from transformers import AutoModel


class SingleBERTWithMLP(nn.Module):
    def __init__(self, config):
        # fmt: off
        super().__init__()
        # fmt: on
        # Extract parameters from config
        model_name = (
            config["models"]["transformer"]["model_name"]
            if not config["is_test"]
            else config["models"]["transformer"]["model_name_test"]
        )
        hidden_size = (
            config["models"]["transformer"]["hidden_size"]
            if not config["is_test"]
            else config["models"]["transformer"]["hidden_size_test"]
        )
        mlp_hidden_size = config["models"]["transformer"]["mlp_hidden_size"]

        # Initialize a single BERT model
        self.bert = AutoModel.from_pretrained(model_name)

        # Define MLP head
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, mlp_hidden_size),  # Double hidden size for concatenation
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, 1),  # Regression output
        )

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
