from transformers import GPT2LMHeadModel, GPT2Tokenizer
from value_model import ValueModel
from reward_model import RewardModel
from ppo_trainer import PPOTrainer
from utils import get_device
import json


def main():
    device = get_device()

    # --- Load Models ---
    model_name = 'gpt2'
    policy_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    ref_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    value_model = ValueModel(model_name).to(device)
    reward_model = RewardModel(model_name).to(device)
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    # --- Load Tokenizer ---
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # --- Load Prompts ---
    with open('data/ppo_prompts.json') as f:
        prompt_data = json.load(f)
    prompts = [item['prompt'] for item in prompt_data]

    # --- create Trainer and Run ---
    trainer = PPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        value_model=value_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        device=device
    )
    trainer.train(prompts)

if __name__ == '__main__':
    main()