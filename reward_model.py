# reward_model.py

import torch
import torch.nn as nn
from transformers import GPT2Model
from tqdm import tqdm
from utils import get_device

class RewardModel(nn.Module):
    def __init__(self, model_name="gpt_2"):
        super().__init__()
        self.backbone = GPT2Model.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state # hidden state of last layer

        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        pooled = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            sequence_lengths
        ] # Hidden state of last non-padding token for each sequence in the batch

        reward = self.reward_head(pooled)
        return reward.squeeze(-1)