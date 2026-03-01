# ppo_trainer.py
from transformers import get_linear_schedule_with_warmup
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from torch.nn.utils import clip_grad_norm_
from utils import get_device

class PPOTrainer:
    def __init__(
        self,
        policy_model: PreTrainedModel,
        ref_model: PreTrainedModel,
        value_model: nn.Module,
        reward_model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        kl_coef: float = 0.1,
        clip_eps: float = 0.2,
        gamma: float = 1.0,
        lam: float = 0.95,
        value_loss_coef: float = 0.5,
        max_gen_len: int = 64
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.value_model = value_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device
        self.kl_coef = kl_coef
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.max_gen_len = max_gen_len



        self.optimizer = torch.optim.AdamW(
            list(self.policy_model.parameters()) + list(self.value_model.parameters()),
            lr=1e-6,
            weight_decay=0.01
        )

    @torch.no_grad() # Because we are generating rolluts (not training) we dont need gradient tracking
    def generate_rollouts(self, prompts: list[str]) -> list[dict]:
        # --- Init ---
        # Set models to eval from train mode.
        self.policy_model.eval()
        self.ref_model.eval()
        self.value_model.eval()
        self.reward_model.eval()

        # --- Left pad abd tokenize inputs ---
        # Left pad the inputs
        self.tokenizer.padding_side = 'left'
        encoded = self.tokenizer(
            prompts,
            return_tensors='pt', # pytorch tensor
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        # encoded['input_ids']:      (batch_size, padded_prompt_len)  — token IDs
        # encoded['attention_mask']:  (batch_size, padded_prompt_len)  — 1 for real tokens, 0 for PAD
        padded_prompt_len = encoded['input_ids'].shape[1]

        # --- Generate Rollout (policy model) ---
        # Batched generation
        gen_output = self.policy_model.generate(
            **encoded,
            max_new_tokens=self.max_gen_len,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        full_ids = gen_output.sequences             # (batch_size, total_len)
        total_len = full_ids.shape[1]
        gen_len = total_len - padded_prompt_len
        gen_ids = full_ids[:, padded_prompt_len:]   # (batch_size, gen_len)

        # --- Build attention mask ---
        # Build attention mask for prompt + gen
        is_eos = (gen_ids == self.tokenizer.eos_token_id)   # (batch_Szie, gen_len)
        eos_cumsum = is_eos.cumsum(dim=1) # all tokens before first eos == 0, all tokens before second eos == 1 , so on
        # (before first eos OR exactly first eos)
        gen_mask = ((eos_cumsum == 0) | (is_eos & (eos_cumsum == 1))).long() # convert to long bc torch uses 64 bit integers
        full_mask = torch.cat([
            encoded['attention_mask'],
            gen_mask,
        ], dim=1)

        # --- Policy model forward pass ---
        # Get log_probs for policy model with forward pass
        policy_logits = self.policy_model(
            input_ids=full_ids,
            attention_mask=full_mask
        ).logits    # (batch_size, total_len, vocab size) 

        #  Note: Although this is somewhat redundant since we could extract the values from scores,
        # We would need to do some annoying transforms, and the forward pass is pretty efficient since we can generate the scores in one shot.
        # --- Extract Logits from get_output ---
        # scores = torch.stack(gen_output.scores, dim=0)     # (gen_len, batch_size, vocab_size)
        # log_probs_all = F.log_softmax(scores, dim=-1)      # (gen_len, batch_size, vocab_size)
        # log_probs_all = log_probs_all.permute(1, 0, 2)     # (batch_size, gen_len, vocab_size)
        # policy_logprobs = log_probs_all.gather(
        #     dim=-1, index=gen_ids.unsqueeze(-1)
        # ).squeeze(-1)   
        # ---------------------------------------

        # logits at pos t predict the token at pos t+1
        # so logits at [padded_prompt_len - 1 : total_len - 1] are our predictors
        policy_logprobs_all = F.log_softmax(policy_logits, dim=-1)                          # (batch_size, total_len, vocab_size)
        policy_logprobs_gen = policy_logprobs_all[:, padded_prompt_len-1:total_len-1, :]    # (batch_size, gen_len, vocab_size)
        policy_logprobs = policy_logprobs_gen.gather(
            dim=-1,
            index=gen_ids.unsqueeze(-1) #  (batch, gen_len, 1) unsqueeze so the dimsensions match
        ).squeeze(-1) # squeeze to drop extra dim
        # Note: gather will index into logprobs for each (batch, postion, index) triplet. in our case we only have 1 index
        # This pulls out the logprob for the chosen token at each postion in the sequence for each batch

        # --- Ref model forward pass ---
        # Get log probs for ref model with forward pass
        ref_logits = self.ref_model(
            input_ids=full_ids,
            attention_mask=full_mask
        ).logits
        ref_logprobs_all = F.log_softmax(ref_logits, dim=-1)
        ref_logprobs_gen = ref_logprobs_all[:, padded_prompt_len-1 : total_len-1, :]
        ref_logprobs = ref_logprobs_gen.gather(
            dim=-1,
            index=gen_ids.unsqueeze(-1)
        ).squeeze(-1)

        # --- Value Estimates ---
        values_all = self.value_model(
            input_ids=full_ids,
            attention_mask=full_mask
        ) # (batch_size, total_len)
        gen_values = values_all[:, padded_prompt_len:] # (batxh_size, gen_len)

        # --- Reward Scores ---
        reward_scores = self.reward_model(
            input_ids=full_ids,
            attention_mask=full_mask
        )   # (batch_size, )

        # --- Per token rewards
        policy_diff_per_gen_token = policy_logprobs - ref_logprobs      # (batch_size, gen_len)
        rewards = -self.kl_coef * policy_diff_per_gen_token    # (batch_size, gen_len) reward at each token starts as just kl scores

        # add reward score to last generated token (not padding)
        last_real_idx = gen_mask.sum(dim=1) - 1 # (batch_size,)
        rewards[
            torch.arange(rewards.shape[0], device=self.device),
            last_real_idx
        ] += reward_scores

        # --- Reset padding to right ---
        self.tokenizer.padding_side = 'right'

        # --- Pack rollouts ---
        rollouts = []
        batch_size = full_ids.shape[0]
        for i in range(batch_size):
            rollouts.append({
                'full_ids': full_ids[i],                # (total_len,)
                'full_mask': full_mask[i],               # (total_len,)
                'gen_ids': gen_ids[i],                   # (gen_len,)
                'gen_mask': gen_mask[i],                  # (gen_len,)
                'old_logprobs': policy_logprobs[i],      # (gen_len,)
                'values': gen_values[i],                  # (gen_len,)
                'rewards': rewards[i],                    # (gen_len,)
                'reward_score': reward_scores[i].item(),  # scalar
                'prompt_text': prompts[i],                # string
                'response_text': self.tokenizer.decode(
                    gen_ids[i][gen_mask[i].bool()],
                    skip_special_tokens=True
                ),
            })
        return rollouts
    
    def compute_advantages(self, rewards, values, mask):
        """
        GAE (Generalized Advantage Estimation)
        rewards: (gen_len,) - per-token rewards 
        values:  (gen_len,) - value model estimates
        mask:    (gen_len,) - attention mask
        returns: (advantages, returns) each (gen_len,) 
        """
        gen_len = rewards.shape[0]
        advantages = torch.zeros(gen_len, device=self.device)
        gae = 0.0

        for t in reversed(range(gen_len)):
            if mask[t] == 0:
                continue

            # V(t+1): next token's value, or 0 if last real token
            next_value = values[t+1] if (t+1 < gen_len and mask[t+1] == 1) else 0.0

            # Temporal Difference (TD) error: actual reward + discounted future value - current value estimate
            delta = rewards[t] + self.gamma * next_value - values[t]

            # GAE accumulation: delta _ dsicounted carried-forward advantage
            gae = delta + self.gamma *self.lam * gae # Reward is passed back through gae variable
            advantages[t] = gae

        returns = advantages + values # (gen_len,)
        return advantages, returns

    def ppo_step(self, rollouts: list[dict], ppo_epochs: int = 4):
        """Run PPO optimization on rollouts"""
        self.policy_model.train()
        self.value_model.train()

        # --- Compute advantages for each rollout ---
        for rollout in rollouts:
            advantages, returns = self.compute_advantages(
                rollout['rewards'],
                rollout['values'],
                rollout['gen_mask']
            )
            rollout['advantages'] = advantages
            rollout['returns'] = returns
        
        # --- PPO epochs: re-use same rollouts multiple times ---
        total_policy_loss = 0.0
        total_value_loss = 0.0
        for epoch in range(ppo_epochs):
            full_ids = torch.stack([r['full_ids'] for r in rollouts])            # (batch_size, total_len)
            full_mask = torch.stack([r['full_mask'] for r in rollouts])          # (batch_size, total_len)
            gen_ids = torch.stack([r['gen_ids'] for r in rollouts])              # (batch_size, gen_len)
            gen_mask = torch.stack([r['gen_mask'] for r in rollouts])            # (batch_size, gen_len)
            old_logprobs = torch.stack([r['old_logprobs'] for r in rollouts])    # (batch_size, gen_len)
            advantages = torch.stack([r['advantages'] for r in rollouts])        # (batch_size, gen_len)
            returns = torch.stack([r['returns'] for r in rollouts])              # (batch_size, gen_len)

            total_len = full_ids.shape[1]
            gen_len = gen_ids.shape[0]
            prompt_len = total_len - gen_len

            # --- recompute policy log-probs ---
            # Forward pass on current policy
            curr_logits = self.policy_model(
                input_ids=full_ids,
                attention_mask=full_mask
            ).logits    # (1, total_len, vocab_len)

            # Same offset + gather logic as in gen_rollouts
            curr_logprobs_all = F.log_softmax(curr_logits, dim =-1)                 # (batch_size, total_len, vocab_Size)
            curr_logprobs_gen = curr_logprobs_all[:, prompt_len-1:total_len-1, :]   # (batch_size, gen_len, vocab_size)
            curr_logprobs = curr_logprobs_gen.gather(
                dim=-1,
                index=gen_ids.unsqueeze(-1) # (batch_size, gen_len, 1)
            ).squeeze(-1)                                                                       # (batch_size, gen_len)

