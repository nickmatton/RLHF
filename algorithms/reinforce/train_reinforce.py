# train_reinforce.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

from common.utils import get_device
from algorithms.reinforce.reinforce_trainer import ReinforceTrainer
from models.reward_model import RewardModel


def main():
    device = get_device()
    model_name = "gpt2"

    # --- Policy: the model we're training ---
    policy_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    # --- Reference: frozen copy to compute KL penalty ---
    ref_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # --- Reward model: trained in rlhf/ Step 2 ---
    reward_model = RewardModel(model_name).to(device)
    reward_model.load_state_dict(
        torch.load("checkpoints/reward/reward_model.pt", map_location=device)
    )
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # --- Prompts ---
    prompts = [
        "Write a short poem about the ocean",
        "Explain what gravity is",
        "Describe a beautiful sunset",
        "Write a haiku about winter",
        "Explain why the sky is blue",
        "Tell me a fun fact about space",
        "Describe your favorite season",
        "Write a limerick about a cat",
        "Explain how rainbows form",
        "Describe what it feels like to fly",
    ]

    # --- Train ---
    trainer = ReinforceTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        device=device,
        lr=1e-5,
        kl_coef=0.1,
        max_gen_len=64,
        batch_size=4,
        num_iterations=50,
    )
    trainer.train(prompts)


if __name__ == "__main__":
    main()