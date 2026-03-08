# Building REINFORCE Policy Gradient from Scratch (with GPT-2)

A hands-on tutorial implementing the REINFORCE algorithm to fine-tune GPT-2 — fully trainable on a MacBook.

---

## What You'll Build

REINFORCE is the simplest policy gradient algorithm and the conceptual foundation for PPO (used in RLHF). By the end of this tutorial, you'll have implemented:

1. **Reward Scoring** — Using the trained reward model from `models/` to score GPT-2 completions
2. **Rollout Generation** — Sampling completions from GPT-2 and computing per-token log-probabilities
3. **The REINFORCE Update** — Computing returns and updating the model with the policy gradient
4. **A KL Penalty** — Preventing the policy from drifting too far from the original model
5. **A Learned Baseline** — Adding a value head to reduce variance

We use **GPT-2 (124M params)** because it fits comfortably in MacBook RAM, trains in minutes, and demonstrates the exact same mechanics used to train much larger models.

---

## The Core Idea

In supervised fine-tuning (SFT), you have labeled data — input/output pairs. But what if you only have a **reward signal**? For example: "this completion is good" or "this completion is bad," with no example of what the "correct" completion should be.

REINFORCE answers this with the **policy gradient theorem**:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

Translated to the language model setting:
- The **policy** $\pi_\theta$ is GPT-2 — it outputs a probability distribution over tokens
- An **action** $a_t$ is choosing a token at position $t$
- A **state** $s_t$ is the sequence of tokens generated so far
- The **return** $G_t$ is the reward signal for the full completion

The update rule: if a completion got high reward, increase the probability of every token in that completion. If it got low reward, decrease them. Over many iterations, the model learns to produce high-reward completions.

---

## Prerequisites & Setup

### Hardware Requirements

- **MacBook with 8GB+ RAM** — sufficient for GPT-2 Small
- **Apple Silicon (M1/M2/M3)** — we'll use MPS acceleration when available
- **~2GB disk space** for model weights

### Environment Setup

```bash
python3 -m venv reinforce-env
source reinforce-env/bin/activate

pip install torch transformers tqdm numpy

# Verify your device
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('MPS (Apple Silicon GPU) available')
elif torch.cuda.is_available():
    print('CUDA available')
else:
    print('CPU only — training will be slower but still works')
"
```

### Prerequisites from the RLHF Tutorial

This tutorial reuses the **trained reward model** from `algorithms/rlhf/`. Make sure you've completed Step 2 of the RLHF tutorial (reward model training) so that `checkpoints/reward/reward_model.pt` exists.

### Project Structure

```
rl-from-scratch/
├── common/utils.py                        # Device selection
├── models/
│   ├── reward_model.py                    # RewardModel (shared across algorithms)
│   └── value_model.py                     # ValueModel (shared across algorithms)
├── algorithms/
│   ├── reinforce/
│   │   ├── reinforce-tutorial.md          # This tutorial
│   │   ├── reinforce_trainer.py           # Core REINFORCE algorithm
│   │   └── train.py                       # Entry point
│   └── rlhf/                             # RLHF pipeline (SFT + Reward + PPO)
├── prompts/
│   └── reinforce_prompt_template.md
├── checkpoints/reward/reward_model.pt     # Trained reward model weights
└── data/
```

---

## Step 0: Shared Utilities

Same device utility used across the project (in `common/utils.py`).

```python
# common/utils.py
import torch

def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")
```

---

## Step 1: The Reward Model

Before we can do RL, we need a reward signal. We already have a trained reward model from the RLHF tutorial — it was trained on human preference data using the Bradley-Terry loss. The exact same reward model used for PPO works for REINFORCE.

### Quick Review: RewardModel Architecture

The reward model (defined in `models/reward_model.py`) is a GPT-2 backbone with a linear reward head:

```python
# From models/reward_model.py
class RewardModel(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.backbone = GPT2Model.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        # Pool from last real token (not padding)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        pooled = hidden_states[torch.arange(input_ids.shape[0], device=hidden_states.device), sequence_lengths]
        reward = self.reward_head(pooled)
        return reward.squeeze(-1)  # (batch_size,)
```

**Key interface:** Give it `(input_ids, attention_mask)` for a full sequence (prompt + completion), get back a scalar reward per sequence. This is identical to how the PPO trainer calls it — the REINFORCE algorithm doesn't care whether the reward comes from a rule-based function or a learned model.

### Loading the Trained Reward Model

```python
import torch
from models.reward_model import RewardModel

model_name = "gpt2"
device = torch.device("mps")  # or "cpu"

reward_model = RewardModel(model_name).to(device)
reward_model.load_state_dict(
    torch.load("checkpoints/reward/reward_model.pt", map_location=device)
)
reward_model.eval()
for param in reward_model.parameters():
    param.requires_grad = False
```

The reward model is frozen during REINFORCE training — we only update the policy. This is the same pattern used in `algorithms/rlhf/train_ppo.py`.

---

## Step 2: Understanding the LLM as a Policy

In classic RL (e.g., CartPole), the policy is a small neural net that outputs a distribution over 2-4 actions. With an LLM, the same logic applies — just at a different scale:

| Classic RL | LLM RL |
|-----------|--------|
| State = 4 floats (cart position, etc.) | State = token sequence so far |
| Action space = 2-4 discrete actions | Action space = 50,257 tokens (GPT-2 vocab) |
| Policy = small MLP | Policy = GPT-2 (124M params) |
| Episode = ~200 steps | Episode = ~64 generated tokens |
| Reward = +1 per step | Reward = scalar score for full completion |

The key operation is the same: sample an action (token) from the policy's distribution, then use the reward to update the policy.

### Extracting Log-Probabilities from GPT-2

In REINFORCE, we need $\log \pi_\theta(a_t | s_t)$ — the log-probability of each generated token under the current policy. GPT-2's forward pass gives us logits at every position, and the logits at position $t$ predict the token at position $t+1$:

```python
# Conceptual illustration — we'll build the real version in Step 3

import torch.nn.functional as F

# Forward pass: get logits for all positions
logits = model(input_ids=full_sequence).logits  # (1, seq_len, vocab_size)

# logits at position t predict token at position t+1
# So for generated tokens starting at prompt_len:
gen_logits = logits[:, prompt_len-1:-1, :]  # (1, gen_len, vocab_size)

# Convert to log-probabilities
log_probs_all = F.log_softmax(gen_logits, dim=-1)  # (1, gen_len, vocab_size)

# Extract log-prob of the specific token that was generated
gen_token_ids = full_sequence[:, prompt_len:]  # (1, gen_len)
log_probs = log_probs_all.gather(
    dim=-1,
    index=gen_token_ids.unsqueeze(-1)  # (1, gen_len, 1)
).squeeze(-1)  # (1, gen_len)
```

**The offset is critical:** `logits[:, t, :]` predicts the token at position `t+1`, not `t`. This off-by-one is the #1 bug in LLM RL implementations.

---

## Step 3: The REINFORCE Trainer

Now we build the full trainer. This combines rollout generation and the policy gradient update.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   REINFORCE Iteration                   │
│                                                         │
│  1. Sample prompts from dataset                         │
│  2. Generate completions (rollout) with policy model    │
│  3. Score completions with reward model                 │
│  4. Compute per-token log-probs via forward pass        │
│  5. Compute KL penalty against reference model          │
│  6. Compute policy gradient loss                        │
│  7. Backprop and update                                 │
└─────────────────────────────────────────────────────────┘
```

```python
# reinforce_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import PreTrainedModel, PreTrainedTokenizer


class ReinforceTrainer:
    def __init__(
        self,
        policy_model: PreTrainedModel,
        ref_model: PreTrainedModel,
        reward_model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        lr: float = 1e-5,
        kl_coef: float = 0.1,
        gamma: float = 1.0,
        max_gen_len: int = 64,
        batch_size: int = 8,
        num_iterations: int = 100,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.max_gen_len = max_gen_len
        self.batch_size = batch_size
        self.num_iterations = num_iterations

        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=lr, weight_decay=0.01
        )

    # ------------------------------------------------------------------
    # Step 3a: Generate rollouts
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_rollouts(self, prompts: list[str]) -> dict:
        """Generate completions and compute everything needed for the update.

        This is the "data collection" phase — we run the policy in the
        environment (= generate text) and record what happened.
        """
        self.policy_model.eval()
        self.ref_model.eval()
        self.reward_model.eval()

        # Left-pad so generated tokens are contiguous at the end
        self.tokenizer.padding_side = "left"
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        padded_prompt_len = encoded["input_ids"].shape[1]

        # --- Sample from the policy ---
        gen_output = self.policy_model.generate(
            **encoded,
            max_new_tokens=self.max_gen_len,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        full_ids = gen_output.sequences  # (batch, total_len)
        total_len = full_ids.shape[1]
        gen_ids = full_ids[:, padded_prompt_len:]  # (batch, gen_len)

        # --- Build generation mask (1 for real tokens, 0 after EOS) ---
        is_eos = gen_ids == self.tokenizer.eos_token_id
        eos_cumsum = is_eos.cumsum(dim=1)
        gen_mask = ((eos_cumsum == 0) | (is_eos & (eos_cumsum == 1))).long()
        full_mask = torch.cat([encoded["attention_mask"], gen_mask], dim=1)

        # --- Policy log-probs (forward pass) ---
        policy_logits = self.policy_model(
            input_ids=full_ids, attention_mask=full_mask
        ).logits
        policy_logprobs = self._extract_gen_logprobs(
            policy_logits, gen_ids, padded_prompt_len, total_len
        )

        # --- Reference model log-probs ---
        ref_logits = self.ref_model(
            input_ids=full_ids, attention_mask=full_mask
        ).logits
        ref_logprobs = self._extract_gen_logprobs(
            ref_logits, gen_ids, padded_prompt_len, total_len
        )

        # --- Score with reward model ---
        # Same interface as rlhf/ppo_trainer.py: pass full sequence, get scalar per sample
        reward_scores = self.reward_model(
            input_ids=full_ids,
            attention_mask=full_mask,
        )  # (batch,)

        # Decode responses for logging
        response_texts = [
            self.tokenizer.decode(gen_ids[i][gen_mask[i].bool()], skip_special_tokens=True)
            for i in range(gen_ids.shape[0])
        ]

        # --- Per-token KL penalty ---
        kl_per_token = self.kl_coef * (policy_logprobs - ref_logprobs)

        self.tokenizer.padding_side = "right"

        return {
            "full_ids": full_ids,
            "full_mask": full_mask,
            "gen_ids": gen_ids,
            "gen_mask": gen_mask,
            "old_logprobs": policy_logprobs,  # (batch, gen_len)
            "kl_per_token": kl_per_token,     # (batch, gen_len)
            "reward_scores": reward_scores,   # (batch,)
            "prompt_texts": prompts,
            "response_texts": response_texts,
        }

    def _extract_gen_logprobs(self, logits, gen_ids, prompt_len, total_len):
        """Extract log-probabilities for the generated tokens.

        logits[:, t, :] predicts token at position t+1, so we offset by -1.
        """
        logprobs_all = F.log_softmax(logits, dim=-1)
        logprobs_gen = logprobs_all[:, prompt_len - 1 : total_len - 1, :]
        return logprobs_gen.gather(
            dim=-1, index=gen_ids.unsqueeze(-1)
        ).squeeze(-1)

    # ------------------------------------------------------------------
    # Step 3b: Monte Carlo returns
    # ------------------------------------------------------------------

    def compute_monte_carlo_returns(self, reward_scores, kl_per_token, gen_mask):
        """Compute Monte Carlo returns from a completed episode.

        REINFORCE uses **Monte Carlo** returns — the actual observed
        cumulative reward from the trajectory, NOT a bootstrapped
        estimate. This is what makes REINFORCE a Monte Carlo method:

        - Monte Carlo (REINFORCE): G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
          Uses real rewards from the full rollout. Unbiased but high variance.

        - Temporal Difference (TD/PPO): G_t = r_t + gamma * V(s_{t+1})
          Bootstraps from a value estimate. Lower variance but biased.

        Reward structure (same pattern as PPO in rlhf/):
        - Every generated token gets a KL penalty: -kl_coef * (log pi - log pi_ref)
        - The final real token also gets the completion-level reward score
        """
        batch_size, gen_len = gen_mask.shape
        rewards = -kl_per_token.clone()  # (batch, gen_len)

        # Add reward score to last real token
        last_real_idx = gen_mask.sum(dim=1) - 1  # (batch,)
        rewards[
            torch.arange(batch_size, device=self.device), last_real_idx
        ] += reward_scores

        # Compute discounted returns (backwards pass)
        returns = torch.zeros_like(rewards)
        for i in range(batch_size):
            G = 0.0
            for t in reversed(range(gen_len)):
                if gen_mask[i, t] == 0:
                    continue
                G = rewards[i, t] + self.gamma * G
                returns[i, t] = G

        return returns

    # ------------------------------------------------------------------
    # Step 3c: The REINFORCE update
    # ------------------------------------------------------------------

    def reinforce_step(self, rollouts: dict):
        """One REINFORCE gradient update.

        Loss = -mean(log_prob(a_t) * G_t)   (over all real tokens in batch)

        This is the policy gradient: actions in high-return completions get
        reinforced, actions in low-return completions get suppressed.
        """
        self.policy_model.train()

        # Recompute log-probs under current policy (same as rollout since
        # we only do one gradient step, but this ensures grad tracking)
        full_ids = rollouts["full_ids"]
        full_mask = rollouts["full_mask"]
        gen_ids = rollouts["gen_ids"]
        gen_mask = rollouts["gen_mask"]

        total_len = full_ids.shape[1]
        gen_len = gen_ids.shape[1]
        prompt_len = total_len - gen_len

        logits = self.policy_model(
            input_ids=full_ids, attention_mask=full_mask
        ).logits
        log_probs = self._extract_gen_logprobs(logits, gen_ids, prompt_len, total_len)

        # Compute returns
        returns = self.compute_monte_carlo_returns(
            rollouts["reward_scores"], rollouts["kl_per_token"], gen_mask
        )

        # Normalize returns across all real tokens in the batch
        mask_bool = gen_mask.bool()
        returns_masked = returns[mask_bool]
        if returns_masked.numel() > 1:
            returns_normalized = (returns_masked - returns_masked.mean()) / (
                returns_masked.std() + 1e-8
            )
        else:
            returns_normalized = returns_masked

        # Policy gradient loss
        log_probs_masked = log_probs[mask_bool]
        policy_loss = -(log_probs_masked * returns_normalized.detach()).mean()

        # Gradient step
        self.optimizer.zero_grad()
        policy_loss.backward()
        clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Logging stats
        with torch.no_grad():
            kl = (rollouts["old_logprobs"][mask_bool] - log_probs[mask_bool].detach()).mean().item()
            avg_reward = rollouts["reward_scores"].mean().item()

        return {
            "policy_loss": policy_loss.item(),
            "mean_kl": kl,
            "avg_reward": avg_reward,
        }

    # ------------------------------------------------------------------
    # Step 3d: Training loop
    # ------------------------------------------------------------------

    def train(self, prompts: list[str]):
        """Main training loop."""

        with open("prompts/reinforce_prompt_template.md", "r") as f:
            template = f.read()

        formatted_prompts = [template.format(prompt=p) for p in prompts]

        for iteration in range(self.num_iterations):
            # Sample a batch of prompts
            batch_indices = torch.randint(0, len(formatted_prompts), (self.batch_size,))
            batch_prompts = [formatted_prompts[i] for i in batch_indices]

            # Generate rollouts and update
            rollouts = self.generate_rollouts(batch_prompts)
            stats = self.reinforce_step(rollouts)

            print(
                f"Iter {iteration + 1}/{self.num_iterations} | "
                f"Loss: {stats['policy_loss']:.4f} | "
                f"KL: {stats['mean_kl']:.4f} | "
                f"Avg Reward: {stats['avg_reward']:.4f}"
            )

            if (iteration + 1) % 10 == 0:
                print(f"  Sample: {rollouts['response_texts'][0][:200]}")

        # Save trained policy
        self.policy_model.save_pretrained("checkpoints/reinforce")
        self.tokenizer.save_pretrained("checkpoints/reinforce")
        print("Saved policy model to checkpoints/reinforce")
```

---

## Step 4: Understanding Each Piece

Let's walk through what's happening in the trainer, because the code has several non-obvious details.

### 4a: Why a Reference Model?

Without constraints, REINFORCE will push the policy wherever rewards are highest — often collapsing to degenerate text that "hacks" the reward function. The **KL penalty** prevents this:

$$r_t = r_{\text{task}} - \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})$$

At each token, we penalize the policy for diverging from the reference (original) model. This keeps generations coherent and grammatical. The `kl_coef` ($\beta$) controls the trade-off — higher values produce more conservative updates.

The reference model is a frozen copy of the policy model at the start of training. Its parameters never change.

### 4b: Per-Token Rewards and Monte Carlo Returns

The reward model scores the **entire completion** — we only get one scalar at the end.

We assign this reward structure:
- **Every token** gets a small KL penalty (stay close to reference)
- **The last real token** also gets the completion-level reward score

Then we compute **Monte Carlo returns** — this is what makes REINFORCE a Monte Carlo method. For each token position $t$, we sum up all the actual observed rewards from $t$ to the end of the episode:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{T-t} r_T$$

This is computed with a simple backwards pass:
```
G_T = r_T
G_{T-1} = r_{T-1} + gamma * G_T
G_{T-2} = r_{T-2} + gamma * G_{T-1}
...
```

**Monte Carlo vs. Temporal Difference (TD):** The alternative to Monte Carlo is TD, which PPO uses via GAE. Instead of summing actual rewards, TD *bootstraps* — it estimates future return using a value function: $G_t \approx r_t + \gamma V(s_{t+1})$. TD has lower variance but introduces bias (the value estimate is wrong early in training). Monte Carlo is unbiased but noisier, which is why REINFORCE benefits so much from the baseline in Step 6.

### 4c: Why Normalize Returns?

Without normalization, if all completions get positive rewards (common early in training), every action gets reinforced — the gradient just says "do more of everything." Normalizing to zero mean ensures roughly half the tokens get positive updates and half get negative, which gives a much more useful gradient signal.

### 4d: The Off-by-One in Log-Prob Extraction

This is worth repeating because it's the most common bug:

```
Position:     0     1     2     3     4     5
Tokens:     [The] [cat] [sat] [on] [the] [mat]
Logits[0] predicts → "cat"
Logits[1] predicts → "sat"
Logits[2] predicts → "on"
...
```

So to get the log-prob of the generated token at position `t`, we use `logits[t-1]`. That's why we slice `logits[:, prompt_len-1 : total_len-1, :]`.

---

## Step 5: Prompt Template and Training Script

### Prompt Template

```markdown
# prompts/reinforce_prompt_template.md
### Instruction: {prompt}
### Response:
```

### Training Script

```python
# algorithms/reinforce/train.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
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

    # --- Reward model: trained in algorithms/rlhf/ Step 2 ---
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
```

### Run It

```bash
# From the repo root:
python algorithms/reinforce/train.py
```

Expected output:
```
Using Apple Silicon GPU (MPS)
Iter  1/50 | Loss: 0.0523 | KL: 0.0012 | Avg Reward: -0.3421
Iter  2/50 | Loss: 0.0481 | KL: 0.0018 | Avg Reward: -0.1253
...
Iter 10/50 | Loss: 0.0312 | KL: 0.0089 | Avg Reward: 0.8734
  Sample: The ocean is a vast and beautiful place, full of wonder and mystery...
...
Iter 50/50 | Loss: 0.0198 | KL: 0.0234 | Avg Reward: 1.4521
Saved policy model to checkpoints/reinforce
```

---

## Step 6: Adding a Learned Baseline (Variance Reduction)

Vanilla REINFORCE has **high variance** — the same prompt can get very different rewards on different rollouts, leading to noisy gradients. The standard fix is to subtract a **baseline** $b(s_t)$:

$$\nabla_\theta J = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))\right]$$

The quantity $G_t - b(s_t)$ is called the **advantage**. It tells us: "was this action better or worse than expected?" Subtracting a baseline doesn't bias the gradient but dramatically reduces variance.

The best baseline is $b(s_t) = V(s_t)$ — the expected return from state $s_t$. We learn this with a **value head** on top of the GPT-2 backbone (same architecture as `models/value_model.py`):

```python
# value_model.py
import torch.nn as nn
from transformers import GPT2Model


class ValueModel(nn.Module):
    """Per-token value estimates V(s_t).

    Uses a frozen GPT-2 backbone with a trainable linear head.
    Architecture matches rlhf/value_model.py.
    """

    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.backbone = GPT2Model.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
        values = self.value_head(hidden_states)     # (batch, seq_len, 1)
        return values.squeeze(-1)                   # (batch, seq_len)
```

### Updated Trainer with Baseline

The changes to the trainer are minimal. In `reinforce_step`, we:

1. Compute value estimates $V(s_t)$ for each generated token
2. Compute advantages: $A_t = G_t - V(s_t)$
3. Use advantages instead of raw returns in the policy loss
4. Add a value loss: $L_V = \text{MSE}(V(s_t), G_t)$

```python
# Key changes to reinforce_trainer.py for baseline version:

# In __init__, add:
self.value_model = value_model  # ValueModel instance
self.value_optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, self.value_model.parameters()),
    lr=1e-5,
    weight_decay=0.01,
)

# In reinforce_step, replace the returns/loss section with:

# Compute returns (same as before)
returns = self.compute_monte_carlo_returns(
    rollouts["reward_scores"], rollouts["kl_per_token"], gen_mask
)

# Compute value estimates
values_all = self.value_model(input_ids=full_ids, attention_mask=full_mask)
gen_values = values_all[:, prompt_len:]  # (batch, gen_len)

# Advantages = Returns - Values
advantages = returns - gen_values.detach()

# Normalize advantages
mask_bool = gen_mask.bool()
adv_masked = advantages[mask_bool]
adv_normalized = (adv_masked - adv_masked.mean()) / (adv_masked.std() + 1e-8)

# Policy loss: use advantages instead of raw returns
log_probs_masked = log_probs[mask_bool]
policy_loss = -(log_probs_masked * adv_normalized.detach()).mean()

# Value loss: teach the value model to predict actual returns
value_loss = F.mse_loss(gen_values[mask_bool], returns[mask_bool].detach())

# Combined backward pass
total_loss = policy_loss + 0.5 * value_loss
self.optimizer.zero_grad()
self.value_optimizer.zero_grad()
total_loss.backward()
clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
clip_grad_norm_(self.value_model.parameters(), max_norm=1.0)
self.optimizer.step()
self.value_optimizer.step()
```

The baseline version will show smoother training curves and converge faster.

---

## Step 7: Debugging Checklist

LLM RL is notoriously tricky to debug. Here are the most common failure modes:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Reward stays flat | Learning rate too low | Try `1e-5` to `5e-5` |
| KL divergence explodes | Learning rate too high or `kl_coef` too low | Reduce LR or increase `kl_coef` |
| Model produces gibberish | KL penalty too weak, policy diverged | Increase `kl_coef` to 0.2+ |
| All completions identical | Temperature too low or model collapsed | Increase temperature, check for mode collapse |
| `NaN` in loss | Numerical instability in log_softmax | Check for zero-length generations |
| Reward goes up but text quality doesn't | Reward model being "hacked" | Increase KL penalty, inspect generated text |

### Key Sanity Checks

```python
# 1. Are log-probs reasonable? (should be negative, typically -1 to -5)
print(f"Mean log-prob: {log_probs[mask_bool].mean().item():.3f}")

# 2. Is KL staying small? (should be < 1.0, ideally < 0.3)
print(f"KL divergence: {kl:.4f}")

# 3. Are gradients flowing?
for name, param in policy_model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item():.6f}")

# 4. Is the generation mask correct?
for i in range(batch_size):
    real_tokens = gen_mask[i].sum().item()
    total_tokens = gen_mask[i].shape[0]
    print(f"Sample {i}: {real_tokens}/{total_tokens} real tokens")
```

---

## How REINFORCE Connects to PPO

REINFORCE is conceptually simple but has two big problems that PPO solves:

### Problem 1: Sample Efficiency

REINFORCE is **on-policy** — each batch of rollouts is used for exactly one gradient step, then discarded. This is wasteful. PPO uses **importance sampling** with a clipped ratio to reuse the same rollouts for multiple gradient steps:

```
REINFORCE: generate → 1 gradient step → discard
PPO:       generate → 4 gradient steps → discard
```

The clipping (`clip_eps=0.2`) prevents the policy from changing too much on reused data, which would invalidate the old log-probs.

### Problem 2: Update Stability

A single REINFORCE update can drastically change the policy (especially with high-variance returns). PPO constrains update size with the clipped surrogate objective:

```python
# REINFORCE loss:
loss = -(log_prob * advantage).mean()

# PPO loss:
ratio = exp(log_prob_new - log_prob_old)
surr1 = ratio * advantage
surr2 = clamp(ratio, 1-eps, 1+eps) * advantage
loss = -min(surr1, surr2).mean()
```

The progression is: **REINFORCE** (this tutorial) → add baseline → add importance sampling + clipping → **PPO** (in `algorithms/rlhf/ppo_trainer.py`). Each step solves a concrete problem with the previous approach.

---

## Summary

The complete REINFORCE-for-LLMs algorithm:

```
Initialize policy model (GPT-2)
Freeze a copy as reference model
Load frozen reward model (from algorithms/rlhf/ training)

For each iteration:
    1. Sample batch of prompts
    2. Generate completions by sampling from policy
    3. Score completions with reward model
    4. Forward pass: compute log-prob of each generated token
    5. Forward pass on ref model: compute KL penalty per token
    6. Build per-token rewards: -kl_penalty + reward_score (at last token)
    7. Compute returns: G_t = discounted sum of future per-token rewards
    8. (Optional) Compute advantages: A_t = G_t - V(s_t)
    9. Normalize returns/advantages
    10. Loss = -mean(log_prob * advantage)
    11. Backprop and update policy (and value model if using baseline)
```

The core is the same as textbook REINFORCE — the LLM-specific complexity is in extracting per-token log-probs (Step 2) and structuring the per-token reward signal (Step 4b). The reward model is reused directly from the RLHF pipeline — once you understand log-prob extraction and per-token reward structure, the rest follows directly.
