# train_reqard.py
import csv
import os
from json import load
from sched import scheduler
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from preference_dataset import PreferenceDataset
from reward_model import RewardModel
from utils import get_device


def train_reward_model(num_epochs=20, batch_size=8, learning_rate=5e-5):
    model_name = "gpt2"
    device = get_device()

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = RewardModel(model_name=model_name)
    model.to(device)
    
    data = load(open("data/reward_model_data.json", 'r'))
    dataset = PreferenceDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Setup training log
    save_path = "checkpoints/reward"
    os.makedirs(save_path, exist_ok=True)
    log_path = f"{save_path}/training_log.csv"
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'loss', 'accuracy'])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.01
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=len(dataloader) * num_epochs
    )

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_pairs = 0

        for batch in dataloader:
            chosen_ids = batch['chosen_ids'].to(device)
            chosen_mask = batch['chosen_mask'].to(device)
            rejected_ids = batch['rejected_ids'].to(device)
            rejected_mask = batch['rejected_mask'].to(device)

            chosen_rewards = model(chosen_ids, chosen_mask)
            rejected_rewards = model(rejected_ids, rejected_mask)

            # Bradlet-Terry loss: -log(sigmoid(chosen_reward - rejected_reward))
            loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_correct += (chosen_rewards > rejected_rewards).sum().item()
            total_pairs += chosen_rewards.shape[0]

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_pairs
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Log to CSV
        log_writer.writerow([epoch + 1, avg_loss, accuracy])
        log_file.flush()
        
    log_file.close()
    print(f"Training log saved to {log_path}")
    
    model_save_path = f"{save_path}/reward_model.pt"
    torch.save(model.state_dict(), model_save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")

if __name__ == "__main__":
    train_reward_model()
