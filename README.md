# RL from Scratch

Implementing reinforcement learning algorithms from scratch for learning purposes. All implementations use GPT-2 (124M params) and run on a MacBook.

## Structure

```
models/              Network architectures (shared across algorithms)
algorithms/
  rlhf/              Full RLHF pipeline: SFT + Reward Model + PPO
  reinforce/         REINFORCE (Monte Carlo policy gradient)
  dpo/               Direct Preference Optimization (planned)
common/              Shared utilities
prompts/             Prompt templates
data/                Training datasets
checkpoints/         Saved model weights
```

## Algorithms

### RLHF (SFT + Reward Model + PPO)

The full RLHF pipeline in three stages:

1. **SFT** - Fine-tune GPT-2 to follow instructions
2. **Reward Model** - Train a model to score responses by human preference
3. **PPO** - Optimize the policy using the reward signal

```bash
# Run all three stages:
python rlhf.py

# Or run stages individually:
python algorithms/rlhf/sft.py
python algorithms/rlhf/train_reward.py
python algorithms/rlhf/train_ppo.py

# Evaluate:
python algorithms/rlhf/test_ppo.py
```

Tutorial: [algorithms/rlhf/rlhf-tutorial.md](algorithms/rlhf/rlhf-tutorial.md)

### REINFORCE

The simplest policy gradient algorithm - uses Monte Carlo returns to update the policy. Conceptual foundation for PPO.

```bash
python algorithms/reinforce/train.py
```

Tutorial: [algorithms/reinforce/reinforce-tutorial.md](algorithms/reinforce/reinforce-tutorial.md)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers tqdm numpy
```
