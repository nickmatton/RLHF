# preference_dataset.py
import torch
from torch.utils.data import Dataset

class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        prompt_path = "prompts/prompt_template.md"
        with open(prompt_path, 'r') as f:
            template = f.read()
            
        self.data = []
        for item in data:
            chosen_prompt = template.format(
                prompt=item['prompt'],
                completion=item['chosen'],
            ) + tokenizer.eos_token
            rejected_prompt = template.format(
                prompt=item['prompt'],
                completion=item['rejected'],
            ) + tokenizer.eos_token

            encoded_chosen = tokenizer(
                chosen_prompt,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            encoded_rejected = tokenizer(
                rejected_prompt,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )

            self.data.append(
                {
                    'chosen_ids': encoded_chosen['input_ids'].squeeze(),
                    'chosen_mask': encoded_chosen['attention_mask'].squeeze(),
                    'rejected_ids': encoded_rejected['input_ids'].squeeze(),
                    'rejected_mask': encoded_rejected['attention_mask'].squeeze(),
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]