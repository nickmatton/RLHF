# Building a Full RLHF Loop from Scratch

A hands-on tutorial using GPT-2 Small (124M parameters) — fully trainable on a MacBook.

---

## What You'll Build

RLHF has three stages. By the end of this tutorial, you'll have implemented all three:

1. **Supervised Fine-Tuning (SFT)** — Teach the base model to follow instructions
2. **Reward Model Training** — Train a model to score responses by human preference
3. **PPO (Proximal Policy Optimization)** — Optimize the SFT model using the reward signal

We use **GPT-2 Small (124M params)** because it fits comfortably in MacBook RAM (even on CPU), trains in minutes per stage, and demonstrates the full RLHF pipeline identically to how it works at scale.

---

## Prerequisites & Setup

### Hardware Requirements

- **MacBook with 8GB+ RAM** — sufficient for GPT-2 Small
- **Apple Silicon (M1/M2/M3)** — we'll use MPS acceleration when available
- **~2GB disk space** for model weights and datasets

### Environment Setup

```bash
# Create a dedicated environment
python3 -m venv rlhf-env
source rlhf-env/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install tqdm numpy pandas

# Verify your device
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('✅ MPS (Apple Silicon GPU) available — training will be fast')
elif torch.cuda.is_available():
    print('✅ CUDA available')
else:
    print('⚠️  CPU only — training will be slower but still works')
"
```

### Project Structure

Create this folder structure:

```
rlhf-from-scratch/
├── step1_sft.py            # Supervised fine-tuning
├── step2_reward_model.py   # Reward model training
├── step3_ppo.py            # PPO training loop
├── utils.py                # Shared utilities
├── data/                   # Datasets (auto-downloaded)
├── checkpoints/            # Saved models
│   ├── sft/
│   ├── reward/
│   └── ppo/
└── evaluate.py             # Test your final model
```

```bash
mkdir -p rlhf-from-scratch/checkpoints/{sft,reward,ppo} rlhf-from-scratch/data
cd rlhf-from-scratch
```

---

## Step 0: Shared Utilities

Create `utils.py` with helpers used across all stages.

```python
# utils.py
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

def get_device():
    """Pick the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_base_model(model_name="gpt2", device=None):
    """Load GPT-2 and its tokenizer."""
    if device is None:
        device = get_device()
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    
    print(f"Loaded {model_name} on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer, device

def generate_text(model, tokenizer, prompt, device, max_new_tokens=64, temperature=0.7):
    """Generate a completion from a prompt."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Decode only the generated part (not the prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)
```

---

## Step 1: Supervised Fine-Tuning (SFT)

**Goal:** Turn base GPT-2 (which just completes text) into a model that follows instructions in a prompt/response format.

### 1.1 — Understanding SFT

Base GPT-2 was trained to predict the next token on internet text. If you give it "What is the capital of France?", it might continue with "What is the capital of Germany? What is..." rather than answering. SFT teaches it the pattern:

```
### Instruction: {question}
### Response: {answer}
```

### 1.2 — The SFT Dataset

We'll use a small instruction-following dataset. For this tutorial, we create a synthetic one to keep things self-contained and fast:

```python
# step1_sft.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import json
import os
from utils import load_base_model, generate_text, get_device

# ──────────────────────────────────────────────
# 1. Create a small instruction-following dataset
# ──────────────────────────────────────────────

SFT_DATA = [
    {"instruction": "What is the capital of France?",
     "response": "The capital of France is Paris."},
    {"instruction": "Explain gravity in simple terms.",
     "response": "Gravity is a force that pulls objects toward each other. The Earth's gravity keeps us on the ground and makes things fall when you drop them."},
    {"instruction": "Write a haiku about the ocean.",
     "response": "Waves crash on the shore,\nSalt air fills the morning breeze,\nSea meets sky as one."},
    {"instruction": "What is 15 multiplied by 7?",
     "response": "15 multiplied by 7 equals 105."},
    {"instruction": "Summarize photosynthesis.",
     "response": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. It occurs in the chloroplasts of plant cells."},
    {"instruction": "Give me a tip for better sleep.",
     "response": "Try to maintain a consistent sleep schedule by going to bed and waking up at the same time every day, even on weekends. This helps regulate your body's internal clock."},
    {"instruction": "What is machine learning?",
     "response": "Machine learning is a branch of artificial intelligence where computers learn patterns from data rather than being explicitly programmed. Models improve their performance as they see more examples."},
    {"instruction": "Name three primary colors.",
     "response": "The three primary colors are red, blue, and yellow."},
    {"instruction": "How does a refrigerator work?",
     "response": "A refrigerator works by circulating a refrigerant fluid through a cycle of compression and expansion. The refrigerant absorbs heat from inside the fridge and releases it outside, keeping the interior cold."},
    {"instruction": "Write a short motivational message.",
     "response": "Every expert was once a beginner. Keep showing up, keep learning, and trust the process. Progress isn't always visible day to day, but it compounds over time."},
    {"instruction": "What causes rain?",
     "response": "Rain forms when water evaporates from the surface, rises as warm moist air, cools at higher altitudes, and condenses into water droplets in clouds. When the droplets become heavy enough, they fall as rain."},
    {"instruction": "Translate 'hello' to Spanish.",
     "response": "'Hello' in Spanish is 'hola'."},
    {"instruction": "What is the largest planet in our solar system?",
     "response": "Jupiter is the largest planet in our solar system. It is a gas giant with a mass more than twice that of all other planets combined."},
    {"instruction": "Give me a healthy breakfast idea.",
     "response": "Try overnight oats: combine rolled oats with yogurt and milk, add berries and a drizzle of honey, and refrigerate overnight. It's nutritious, filling, and requires no morning cooking."},
    {"instruction": "Explain what an API is.",
     "response": "An API (Application Programming Interface) is a set of rules that lets different software programs communicate with each other. It defines how to request data or actions from a service without needing to know its internal workings."},
    {"instruction": "What is the boiling point of water?",
     "response": "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."},
]

PROMPT_TEMPLATE = "### Instruction: {instruction}\n### Response: {response}"

class SFTDataset(Dataset):
    """Format instruction/response pairs for causal language model training."""
    
    def __init__(self, data, tokenizer, max_length=256):
        self.examples = []
        for item in data:
            text = PROMPT_TEMPLATE.format(**item) + tokenizer.eos_token
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.examples.append({
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(),
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


# ──────────────────────────────────────────────
# 2. Training loop
# ──────────────────────────────────────────────

def train_sft():
    device = get_device()
    model, tokenizer, device = load_base_model("gpt2", device)
    
    dataset = SFTDataset(SFT_DATA, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    num_epochs = 20  # Small dataset → more epochs
    total_steps = num_epochs * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=10, num_training_steps=total_steps
    )
    
    print(f"\n{'='*50}")
    print(f"SFT Training: {num_epochs} epochs, {len(dataset)} examples")
    print(f"{'='*50}\n")
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # For causal LM, labels = input_ids (shifted internally by the model)
            # We mask padding tokens in labels with -100 so they don't contribute to loss
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} — Loss: {avg_loss:.4f}")
    
    # Save
    save_path = "checkpoints/sft"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n✅ SFT model saved to {save_path}")
    
    # Test it
    print(f"\n{'='*50}")
    print("Testing SFT model:")
    print(f"{'='*50}\n")
    
    test_prompts = [
        "### Instruction: What is the speed of light?\n### Response:",
        "### Instruction: Write a fun fact about dolphins.\n### Response:",
        "### Instruction: What is Python?\n### Response:",
    ]
    for prompt in test_prompts:
        response = generate_text(model, tokenizer, prompt, device, max_new_tokens=80)
        print(f"Prompt: {prompt.split('Instruction: ')[1].split(chr(10))[0]}")
        print(f"Response: {response.strip()}\n")

if __name__ == "__main__":
    train_sft()
```

### 1.3 — Run SFT

```bash
python step1_sft.py
```

**Expected output:** Loss should drop from ~3–4 down to ~0.5–1.0 over 20 epochs. Test generations should show the model has learned the instruction-response format, though quality will be limited (it's a tiny dataset — that's fine, the point is to learn the pattern).

**Key concepts demonstrated:**
- We formatted data as `### Instruction: ... ### Response: ...` — this is the "chat template"
- Labels are the same as inputs (causal LM), with padding masked to `-100`
- The model learns to complete the `### Response:` portion given an instruction

---

## Step 2: Reward Model Training

**Goal:** Train a model that takes a prompt + response and outputs a scalar score indicating how "good" the response is. This replaces human judgment in the RL loop.

### 2.1 — Understanding Reward Models

A reward model is trained on **preference pairs**: given the same prompt, a human has judged response A as better than response B. The model learns to assign higher scores to preferred responses.

The loss function is simple — for a preferred response $y_w$ and a rejected response $y_l$:

$$\mathcal{L} = -\log\sigma(r(x, y_w) - r(x, y_l))$$

This pushes the reward of the preferred response above the rejected one.

### 2.2 — The Preference Dataset

```python
# step2_reward_model.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Tokenizer
from tqdm import tqdm
from utils import get_device

# ──────────────────────────────────────────────
# 1. Preference data: (prompt, chosen, rejected)
# ──────────────────────────────────────────────

PREFERENCE_DATA = [
    {
        "prompt": "Explain what a neural network is.",
        "chosen": "A neural network is a computing system inspired by the brain. It consists of layers of interconnected nodes that process information, learn patterns from data, and make predictions. Each connection has a weight that adjusts during training.",
        "rejected": "Neural networks are complicated math stuff used in AI. They do things with numbers and layers. It's really complex and hard to explain simply."
    },
    {
        "prompt": "What should I do if I feel stressed?",
        "chosen": "Try some deep breathing exercises: inhale slowly for 4 counts, hold for 4, and exhale for 4. Physical activity, even a short walk, can help too. Consider talking to someone you trust about what's bothering you.",
        "rejected": "Just don't be stressed. It's all in your head. You should try to think positive thoughts and everything will be fine."
    },
    {
        "prompt": "How do computers store information?",
        "chosen": "Computers store information as binary data — sequences of 0s and 1s. At the hardware level, this is represented by tiny electrical charges in transistors. Storage devices like SSDs use flash memory cells, while RAM provides fast temporary storage for active tasks.",
        "rejected": "Computers store stuff in their memory. There are different types of memory and they hold your files and programs. It uses electricity to remember things."
    },
    {
        "prompt": "Write a short poem about autumn.",
        "chosen": "Golden leaves descend in silence,\nCrisp air carries woodsmoke traces,\nNature paints in amber hues,\nAs autumn fills the empty spaces.",
        "rejected": "Leaves fall down, it gets cold.\nAutumn is here, I've been told.\nThe end."
    },
    {
        "prompt": "What is the difference between weather and climate?",
        "chosen": "Weather refers to short-term atmospheric conditions in a specific place — like today's temperature or whether it's raining. Climate describes the average weather patterns of a region over long periods, typically 30 years or more. You can think of climate as the personality and weather as the mood.",
        "rejected": "Weather is what's happening outside right now. Climate is the big picture weather. They're related but different timescales."
    },
    {
        "prompt": "Give me advice for a job interview.",
        "chosen": "Research the company thoroughly before you go. Prepare specific examples from your experience that demonstrate relevant skills. Practice common questions out loud. Arrive 10 minutes early, dress appropriately, and prepare thoughtful questions to ask the interviewer.",
        "rejected": "Just be yourself and wing it. If they like you, they like you. Wear something nice I guess."
    },
    {
        "prompt": "Explain how vaccines work.",
        "chosen": "Vaccines introduce a harmless piece of a pathogen — such as a weakened virus or a protein fragment — to your immune system. Your body learns to recognize it and produces antibodies. If you later encounter the real pathogen, your immune system can respond quickly and effectively.",
        "rejected": "Vaccines put a little bit of the disease in you so your body can fight it later. They've been around for a long time and most people get them as kids."
    },
    {
        "prompt": "Suggest a beginner workout routine.",
        "chosen": "Start with 3 days per week. Each session: 5 minutes of walking to warm up, then 3 sets of 10 bodyweight squats, 3 sets of 5 push-ups (knee push-ups are fine), a 30-second plank, and 5 minutes of stretching. Increase reps gradually as you get stronger.",
        "rejected": "Just go to the gym and do some exercises. Start with whatever machines look interesting. Try to go every day if you can."
    },
    {
        "prompt": "What is inflation?",
        "chosen": "Inflation is the rate at which the general level of prices for goods and services rises over time, reducing purchasing power. If inflation is 3%, something that cost $100 last year now costs $103. Central banks try to manage it through monetary policy, typically targeting around 2% annually.",
        "rejected": "Inflation is when prices go up. It's bad because things cost more. The government prints too much money and that causes it."
    },
    {
        "prompt": "How can I improve my writing?",
        "chosen": "Read widely and actively — notice how skilled authors structure sentences and arguments. Write regularly, even just 15 minutes a day. After drafting, revise ruthlessly: cut unnecessary words, replace vague language with specifics, and read your work aloud to catch awkward phrasing.",
        "rejected": "Practice more and read books. Use a thesaurus to find bigger words. Eventually you'll get better at it."
    },
    {
        "prompt": "Why is the sky blue?",
        "chosen": "Sunlight contains all colors of the spectrum. When it enters Earth's atmosphere, shorter blue wavelengths scatter more than other colors when they collide with air molecules — this is called Rayleigh scattering. This scattered blue light reaches our eyes from all directions, making the sky appear blue.",
        "rejected": "The sky is blue because of the way light works in the atmosphere. It has to do with wavelengths and scattering. It's a physics thing."
    },
    {
        "prompt": "What's a good way to save money?",
        "chosen": "Start by tracking all your spending for a month to see where your money goes. Then set up automatic transfers to a savings account on payday — even $50 a month adds up. Look for your biggest recurring expenses (subscriptions, dining out) and see which you can reduce.",
        "rejected": "Spend less than you earn. Try to save some money each month. Avoid buying things you don't need."
    },
]


# ──────────────────────────────────────────────
# 2. Reward model architecture
# ──────────────────────────────────────────────

class RewardModel(nn.Module):
    """
    A reward model built on top of GPT-2.
    
    Takes text input, processes it through GPT-2's transformer layers,
    and projects the final hidden state to a single scalar reward score.
    """
    
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.backbone = GPT2Model.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size  # 768 for gpt2
        
        # Project the final hidden state to a scalar reward
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
        
        # Use the last non-padding token's hidden state as the sequence representation
        # (similar to how [CLS] works in BERT, but at the end for causal models)
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1  # index of last real token
        pooled = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            sequence_lengths,
        ]
        
        reward = self.reward_head(pooled)  # (batch, 1)
        return reward.squeeze(-1)  # (batch,)


# ──────────────────────────────────────────────
# 3. Preference dataset
# ──────────────────────────────────────────────

class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.pairs = []
        prompt_template = "### Instruction: {prompt}\n### Response: {response}"
        
        for item in data:
            chosen_text = prompt_template.format(
                prompt=item["prompt"], response=item["chosen"]
            )
            rejected_text = prompt_template.format(
                prompt=item["prompt"], response=item["rejected"]
            )
            
            chosen_enc = tokenizer(
                chosen_text, truncation=True, max_length=max_length,
                padding="max_length", return_tensors="pt"
            )
            rejected_enc = tokenizer(
                rejected_text, truncation=True, max_length=max_length,
                padding="max_length", return_tensors="pt"
            )
            
            self.pairs.append({
                "chosen_ids": chosen_enc["input_ids"].squeeze(),
                "chosen_mask": chosen_enc["attention_mask"].squeeze(),
                "rejected_ids": rejected_enc["input_ids"].squeeze(),
                "rejected_mask": rejected_enc["attention_mask"].squeeze(),
            })
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]


# ──────────────────────────────────────────────
# 4. Training loop
# ──────────────────────────────────────────────

def train_reward_model():
    device = get_device()
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = RewardModel("gpt2").to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Reward model parameters: {param_count:,}")
    
    dataset = PreferenceDataset(PREFERENCE_DATA, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    num_epochs = 15
    
    print(f"\n{'='*50}")
    print(f"Reward Model Training: {num_epochs} epochs, {len(dataset)} preference pairs")
    print(f"{'='*50}\n")
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_pairs = 0
        
        for batch in dataloader:
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)
            
            # Get reward scores for both
            reward_chosen = model(chosen_ids, chosen_mask)
            reward_rejected = model(rejected_ids, rejected_mask)
            
            # Bradley-Terry preference loss:
            # We want reward_chosen > reward_rejected
            # Loss = -log(sigmoid(reward_chosen - reward_rejected))
            loss = -torch.nn.functional.logsigmoid(
                reward_chosen - reward_rejected
            ).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            # Track accuracy: how often does the model rank correctly?
            total_correct += (reward_chosen > reward_rejected).float().sum().item()
            total_pairs += chosen_ids.shape[0]
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_pairs
        
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} — Loss: {avg_loss:.4f}, Accuracy: {accuracy:.1%}")
    
    # Save
    save_path = "checkpoints/reward"
    torch.save(model.state_dict(), os.path.join(save_path, "reward_model.pt"))
    tokenizer.save_pretrained(save_path)
    print(f"\n✅ Reward model saved to {save_path}")
    
    # Test: score some responses
    print(f"\n{'='*50}")
    print("Testing reward model scores:")
    print(f"{'='*50}\n")
    
    model.eval()
    test_pairs = [
        {
            "prompt": "What is 2+2?",
            "good": "2 plus 2 equals 4.",
            "bad": "Idk maybe like 5 or something."
        },
        {
            "prompt": "How do airplanes fly?",
            "good": "Airplanes fly because their wings are shaped to create lift. As the plane moves forward, air flows faster over the curved top of the wing than underneath, creating lower pressure above and higher pressure below, which pushes the wing upward.",
            "bad": "They just do. Engines go brrrr and they go up."
        },
    ]
    
    prompt_template = "### Instruction: {prompt}\n### Response: {response}"
    with torch.no_grad():
        for pair in test_pairs:
            good_text = prompt_template.format(prompt=pair["prompt"], response=pair["good"])
            bad_text = prompt_template.format(prompt=pair["prompt"], response=pair["bad"])
            
            good_enc = tokenizer(good_text, return_tensors="pt", truncation=True, max_length=256, padding="max_length").to(device)
            bad_enc = tokenizer(bad_text, return_tensors="pt", truncation=True, max_length=256, padding="max_length").to(device)
            
            good_reward = model(good_enc["input_ids"], good_enc["attention_mask"]).item()
            bad_reward = model(bad_enc["input_ids"], bad_enc["attention_mask"]).item()
            
            print(f"Prompt: {pair['prompt']}")
            print(f"  ✅ Good response reward: {good_reward:.3f}")
            print(f"  ❌ Bad response reward:  {bad_reward:.3f}")
            print(f"  Correct ranking: {'Yes ✓' if good_reward > bad_reward else 'No ✗'}\n")


import os
if __name__ == "__main__":
    train_reward_model()
```

### 2.3 — Run Reward Model Training

```bash
python step2_reward_model.py
```

**Expected output:** Accuracy should reach 90–100% on the training set within 15 epochs. The test examples should show correct ranking (good response gets a higher reward score than bad response).

---

## Step 3: PPO — The RL Loop

**Goal:** Use the reward model to improve the SFT model's responses via reinforcement learning.

### 3.1 — Understanding PPO for Language Models

PPO (Proximal Policy Optimization) treats the language model as a **policy** that takes actions (generating tokens). The "environment" gives a reward (from the reward model) after the full response is generated.

Key components:
- **Policy model** — the model being trained (initialized from SFT checkpoint)
- **Reference model** — a frozen copy of the SFT model, used to compute a KL penalty that prevents the policy from diverging too far
- **Reward model** — the model trained in Step 2, used to score generated responses
- **Value model** — estimates expected future reward, used to compute advantages

The PPO objective is:

$$\mathcal{L}_{\text{PPO}} = -\min\left(\frac{\pi_\theta}{\pi_{\text{old}}} A, \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1-\epsilon, 1+\epsilon\right) A\right) + \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

Where:
- $\pi_\theta / \pi_{\text{old}}$ is the probability ratio between the updated and rollout policies
- $A$ is the advantage (how much better this action was than expected)
- $\epsilon$ clips the ratio to prevent large updates
- $\beta \cdot D_{\text{KL}}$ penalizes drifting too far from the reference model

### 3.2 — Full PPO Implementation

```python
# step3_ppo.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Model, GPT2Tokenizer
from tqdm import tqdm
import copy
import os
from utils import get_device, generate_text

# ──────────────────────────────────────────────
# 1. Value model (critic)
# ──────────────────────────────────────────────

class ValueModel(nn.Module):
    """Estimates expected reward for a given state (token sequence)."""
    
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
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, seq, hidden)
        values = self.value_head(hidden).squeeze(-1)  # (batch, seq)
        return values


# ──────────────────────────────────────────────
# 2. Reward model (reload from step 2)
# ──────────────────────────────────────────────

class RewardModel(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.backbone = GPT2Model.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        batch_size = input_ids.shape[0]
        seq_lengths = attention_mask.sum(dim=1) - 1
        pooled = hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lengths]
        return self.reward_head(pooled).squeeze(-1)


# ──────────────────────────────────────────────
# 3. PPO Trainer
# ──────────────────────────────────────────────

class PPOTrainer:
    def __init__(
        self,
        policy_model,
        ref_model,
        value_model,
        reward_model,
        tokenizer,
        device,
        lr=1e-6,
        kl_coef=0.1,
        clip_eps=0.2,
        gamma=1.0,
        lam=0.95,
        value_loss_coef=0.5,
        max_gen_len=64,
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.value_model = value_model
        self.reward_fn = reward_model
        self.tokenizer = tokenizer
        self.device = device
        
        self.kl_coef = kl_coef
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.max_gen_len = max_gen_len
        
        # Optimize both policy and value model
        self.optimizer = torch.optim.AdamW(
            list(self.policy.parameters()) + list(self.value_model.parameters()),
            lr=lr,
            weight_decay=0.01,
        )
    
    @torch.no_grad()
    def generate_rollouts(self, prompts):
        """
        Generate responses from the policy model and compute:
        - token log-probs under the policy
        - token log-probs under the reference model
        - reward scores
        - value estimates
        """
        self.policy.eval()
        self.ref.eval()
        self.value_model.eval()
        self.reward_fn.eval()
        
        all_data = []
        
        for prompt_text in prompts:
            # Tokenize the prompt
            prompt_enc = self.tokenizer(
                prompt_text, return_tensors="pt"
            ).to(self.device)
            prompt_ids = prompt_enc["input_ids"]
            prompt_len = prompt_ids.shape[1]
            
            # Generate from policy
            gen_output = self.policy.generate(
                **prompt_enc,
                max_new_tokens=self.max_gen_len,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            
            full_ids = gen_output.sequences[0]  # (total_len,)
            gen_ids = full_ids[prompt_len:]      # just the generated tokens
            gen_len = len(gen_ids)
            
            if gen_len == 0:
                continue
            
            full_ids_batch = full_ids.unsqueeze(0)  # (1, total_len)
            attn_mask = torch.ones_like(full_ids_batch)
            
            # Get policy log-probs for generated tokens
            policy_logits = self.policy(
                input_ids=full_ids_batch, attention_mask=attn_mask
            ).logits[0]  # (total_len, vocab)
            
            # Log-prob of each generated token: logits at position t predict token t+1
            policy_logprobs = F.log_softmax(policy_logits, dim=-1)
            gen_logprobs = policy_logprobs[prompt_len - 1 : prompt_len - 1 + gen_len]
            gen_logprobs = gen_logprobs[
                torch.arange(gen_len), gen_ids
            ]  # (gen_len,)
            
            # Reference model log-probs
            ref_logits = self.ref(
                input_ids=full_ids_batch, attention_mask=attn_mask
            ).logits[0]
            ref_logprobs = F.log_softmax(ref_logits, dim=-1)
            ref_gen_logprobs = ref_logprobs[prompt_len - 1 : prompt_len - 1 + gen_len]
            ref_gen_logprobs = ref_gen_logprobs[
                torch.arange(gen_len), gen_ids
            ]  # (gen_len,)
            
            # Value estimates for each generated token position
            values = self.value_model(
                input_ids=full_ids_batch, attention_mask=attn_mask
            )[0]  # (total_len,)
            gen_values = values[prompt_len : prompt_len + gen_len]  # (gen_len,)
            
            # Reward: score the full sequence
            reward_score = self.reward_fn(
                input_ids=full_ids_batch, attention_mask=attn_mask
            ).item()
            
            # KL penalty per token
            kl_per_token = gen_logprobs - ref_gen_logprobs  # (gen_len,)
            
            # Build per-token rewards:
            # - KL penalty at every token
            # - The reward model's score applied at the final token
            rewards = -self.kl_coef * kl_per_token  # (gen_len,)
            rewards[-1] += reward_score  # Add the reward model score to the last token
            
            decoded_response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            all_data.append({
                "prompt_text": prompt_text,
                "response_text": decoded_response,
                "full_ids": full_ids_batch,
                "attn_mask": attn_mask,
                "prompt_len": prompt_len,
                "gen_ids": gen_ids,
                "gen_len": gen_len,
                "old_logprobs": gen_logprobs.detach(),
                "ref_logprobs": ref_gen_logprobs.detach(),
                "values": gen_values.detach(),
                "rewards": rewards.detach(),
                "reward_score": reward_score,
            })
        
        return all_data
    
    def compute_advantages(self, rewards, values):
        """Compute GAE (Generalized Advantage Estimation)."""
        gen_len = len(rewards)
        advantages = torch.zeros(gen_len, device=self.device)
        last_gae = 0
        
        for t in reversed(range(gen_len)):
            if t == gen_len - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def ppo_step(self, rollout_data, ppo_epochs=4):
        """Run PPO optimization on collected rollouts."""
        self.policy.train()
        self.value_model.train()
        
        total_policy_loss = 0
        total_value_loss = 0
        total_kl = 0
        
        for _ in range(ppo_epochs):
            for rollout in rollout_data:
                full_ids = rollout["full_ids"]
                attn_mask = rollout["attn_mask"]
                prompt_len = rollout["prompt_len"]
                gen_ids = rollout["gen_ids"]
                gen_len = rollout["gen_len"]
                old_logprobs = rollout["old_logprobs"]
                rewards = rollout["rewards"]
                old_values = rollout["values"]
                
                # Compute advantages
                advantages, returns = self.compute_advantages(rewards, old_values)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Current policy log-probs
                logits = self.policy(
                    input_ids=full_ids, attention_mask=attn_mask
                ).logits[0]
                logprobs = F.log_softmax(logits, dim=-1)
                curr_logprobs = logprobs[prompt_len - 1 : prompt_len - 1 + gen_len]
                curr_logprobs = curr_logprobs[
                    torch.arange(gen_len, device=self.device), gen_ids
                ]
                
                # Current value estimates
                curr_values = self.value_model(
                    input_ids=full_ids, attention_mask=attn_mask
                )[0][prompt_len : prompt_len + gen_len]
                
                # PPO clipped objective
                ratio = (curr_logprobs - old_logprobs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(curr_values, returns)
                
                # Combined loss
                loss = policy_loss + self.value_loss_coef * value_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_model.parameters()),
                    1.0,
                )
                self.optimizer.step()
                
                # Track KL divergence
                with torch.no_grad():
                    kl = (old_logprobs - curr_logprobs).mean().item()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_kl += kl
        
        n = len(rollout_data) * ppo_epochs
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "mean_kl": total_kl / n,
        }


# ──────────────────────────────────────────────
# 4. Main PPO training loop
# ──────────────────────────────────────────────

def main():
    device = get_device()
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("checkpoints/sft")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load SFT model as policy
    print("Loading SFT model as policy...")
    policy = GPT2LMHeadModel.from_pretrained("checkpoints/sft").to(device)
    
    # Frozen copy as reference
    print("Creating frozen reference model...")
    ref_model = GPT2LMHeadModel.from_pretrained("checkpoints/sft").to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    
    # Value model
    print("Initializing value model...")
    value_model = ValueModel("gpt2").to(device)
    
    # Load reward model
    print("Loading reward model...")
    reward_model = RewardModel("gpt2").to(device)
    reward_model.load_state_dict(
        torch.load("checkpoints/reward/reward_model.pt", map_location=device)
    )
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False
    
    # Training prompts — these are the queries the model will learn to answer well
    training_prompts = [
        "### Instruction: Explain what an atom is.\n### Response:",
        "### Instruction: How can I learn a new language?\n### Response:",
        "### Instruction: What are the benefits of exercise?\n### Response:",
        "### Instruction: Describe how a car engine works.\n### Response:",
        "### Instruction: What is the internet?\n### Response:",
        "### Instruction: Give me tips for public speaking.\n### Response:",
        "### Instruction: What causes earthquakes?\n### Response:",
        "### Instruction: How does the stock market work?\n### Response:",
        "### Instruction: What is evolution?\n### Response:",
        "### Instruction: Suggest ways to reduce stress.\n### Response:",
        "### Instruction: What is climate change?\n### Response:",
        "### Instruction: How do I write a good essay?\n### Response:",
    ]
    
    trainer = PPOTrainer(
        policy_model=policy,
        ref_model=ref_model,
        value_model=value_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        device=device,
        lr=1e-6,
        kl_coef=0.1,
        clip_eps=0.2,
        max_gen_len=80,
    )
    
    num_iterations = 30
    batch_size = 4  # Prompts per iteration
    
    print(f"\n{'='*60}")
    print(f"PPO Training: {num_iterations} iterations, {batch_size} prompts each")
    print(f"{'='*60}\n")
    
    for iteration in range(num_iterations):
        # Sample a batch of prompts
        indices = torch.randint(0, len(training_prompts), (batch_size,))
        batch_prompts = [training_prompts[i] for i in indices]
        
        # Generate rollouts
        rollout_data = trainer.generate_rollouts(batch_prompts)
        
        if len(rollout_data) == 0:
            print(f"Iteration {iteration+1}: No valid rollouts, skipping")
            continue
        
        # PPO update
        stats = trainer.ppo_step(rollout_data, ppo_epochs=4)
        
        # Compute mean reward for logging
        mean_reward = sum(r["reward_score"] for r in rollout_data) / len(rollout_data)
        
        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(
                f"Iter {iteration+1:3d}/{num_iterations} | "
                f"Reward: {mean_reward:+.3f} | "
                f"Policy Loss: {stats['policy_loss']:.4f} | "
                f"Value Loss: {stats['value_loss']:.4f} | "
                f"KL: {stats['mean_kl']:.4f}"
            )
            
            # Show a sample generation
            sample = rollout_data[0]
            print(f"  Sample prompt: {sample['prompt_text'][:60]}...")
            print(f"  Sample response: {sample['response_text'][:120]}")
            print()
    
    # Save
    save_path = "checkpoints/ppo"
    policy.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n✅ PPO-trained model saved to {save_path}")
    
    # Final comparison
    print(f"\n{'='*60}")
    print("Final Comparison: SFT vs PPO")
    print(f"{'='*60}\n")
    
    sft_model = GPT2LMHeadModel.from_pretrained("checkpoints/sft").to(device)
    
    eval_prompts = [
        "### Instruction: What is artificial intelligence?\n### Response:",
        "### Instruction: How can I be more productive?\n### Response:",
        "### Instruction: Explain what DNA is.\n### Response:",
    ]
    
    for prompt in eval_prompts:
        q = prompt.split("Instruction: ")[1].split("\n")[0]
        print(f"Question: {q}\n")
        
        sft_response = generate_text(sft_model, tokenizer, prompt, device, max_new_tokens=100)
        ppo_response = generate_text(policy, tokenizer, prompt, device, max_new_tokens=100)
        
        print(f"  SFT response:  {sft_response.strip()[:200]}")
        print(f"  PPO response:  {ppo_response.strip()[:200]}")
        
        # Score both
        reward_model.eval()
        with torch.no_grad():
            for label, response in [("SFT", sft_response), ("PPO", ppo_response)]:
                full_text = prompt + response
                enc = tokenizer(
                    full_text, return_tensors="pt", truncation=True,
                    max_length=256, padding="max_length"
                ).to(device)
                score = reward_model(enc["input_ids"], enc["attention_mask"]).item()
                print(f"  {label} reward score: {score:.3f}")
        print()


if __name__ == "__main__":
    main()
```

### 3.3 — Run PPO

```bash
python step3_ppo.py
```

**Expected output:** Over 30 iterations, you should see:
- Mean reward gradually increasing
- KL divergence staying moderate (not exploding)
- In the final comparison, PPO responses should score higher than SFT responses on the reward model

---

## Step 4: Evaluate & Compare All Three Stages

```python
# evaluate.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from step2_reward_model import RewardModel
from utils import get_device, generate_text

def main():
    device = get_device()
    
    # Load all three models
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading models...")
    base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    sft_model = GPT2LMHeadModel.from_pretrained("checkpoints/sft").to(device)
    ppo_model = GPT2LMHeadModel.from_pretrained("checkpoints/ppo").to(device)
    
    reward_model = RewardModel("gpt2").to(device)
    reward_model.load_state_dict(
        torch.load("checkpoints/reward/reward_model.pt", map_location=device)
    )
    reward_model.eval()
    
    models = {
        "Base GPT-2": base_model,
        "After SFT": sft_model,
        "After RLHF": ppo_model,
    }
    
    eval_prompts = [
        "### Instruction: What causes the seasons to change?\n### Response:",
        "### Instruction: Give advice for someone starting to learn programming.\n### Response:",
        "### Instruction: What is photosynthesis?\n### Response:",
        "### Instruction: How does WiFi work?\n### Response:",
    ]
    
    print(f"\n{'='*70}")
    print("FULL RLHF PIPELINE COMPARISON")
    print(f"{'='*70}")
    
    for prompt in eval_prompts:
        question = prompt.split("Instruction: ")[1].split("\n")[0]
        print(f"\n{'─'*70}")
        print(f"Question: {question}")
        print(f"{'─'*70}")
        
        for name, model in models.items():
            response = generate_text(model, tokenizer, prompt, device, max_new_tokens=100)
            
            # Score it
            full_text = prompt + response
            enc = tokenizer(
                full_text, return_tensors="pt", truncation=True,
                max_length=256, padding="max_length"
            ).to(device)
            with torch.no_grad():
                score = reward_model(enc["input_ids"], enc["attention_mask"]).item()
            
            print(f"\n  [{name}] (reward: {score:+.3f})")
            print(f"  {response.strip()[:250]}")
    
    print(f"\n{'='*70}")
    print("Done! You've built a complete RLHF pipeline from scratch.")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
```

```bash
python evaluate.py
```

---

## Summary of the Full Pipeline

| Stage | What It Does | Input | Output |
|-------|-------------|-------|--------|
| **SFT** | Teaches instruction-following format | Base GPT-2 + instruction/response pairs | Model that can answer questions |
| **Reward Model** | Learns human preferences | Preference pairs (chosen vs rejected) | Scalar score for any response |
| **PPO** | Optimizes via RL against the reward | SFT model + reward model + prompts | Model that generates higher-reward responses |

## Key Takeaways

1. **SFT gets you format compliance.** The model learns "when given an instruction, produce a response" rather than just continuing text.

2. **The reward model encodes quality judgments.** It learns that detailed, accurate, well-structured responses score higher than vague, dismissive ones.

3. **PPO closes the loop.** The model actively *generates* responses, gets scored, and updates itself to produce better responses — this is the reinforcement learning part.

4. **KL divergence is your guardrail.** Without it, the model would "hack" the reward model by producing degenerate text that scores high but is nonsensical. The KL penalty keeps it close to the SFT model.

5. **Scale is the only difference.** This exact pipeline — SFT → Reward Model → PPO — is what companies use to train production chat models. The only differences are dataset size (millions vs. our dozens), model size (billions vs. 124M), and compute (GPU clusters vs. your MacBook).

## What to Explore Next

- **Swap in a real dataset:** Try `HuggingFaceFW/fineweb-edu` for SFT or `Anthropic/hh-rlhf` for preference data
- **Try DPO:** Direct Preference Optimization skips the reward model and PPO entirely — it directly optimizes the policy using preference pairs. It's simpler and often competitive
- **Scale up the model:** Try `gpt2-medium` (355M) or `TinyLlama-1.1B` if you have 16GB+ RAM
- **Add LoRA:** Use parameter-efficient fine-tuning with `peft` to train larger models with less memory
