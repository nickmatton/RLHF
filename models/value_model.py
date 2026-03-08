# value_model.py
import torch
import torch.nn as nn
from transformers import GPT2Model

class ValueModel(nn.Module):
    def __init__(self, model_name="gpt_2"):
        super().__init__()
        self.backbone = GPT2Model.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        ) 
        hidden_states = outputs.last_hidden_state # (batch_size, seq_len, hidden_size)
        values = self.value_head(hidden_states)   # (batch_size, seq_len, 1) no pooling, we want a value for each token
        return values.squeeze(-1) # (batch_size, seq_len)