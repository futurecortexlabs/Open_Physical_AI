"""Clipped PPO in PyTorch, with explicit timeout bootstrap and rollout boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass
class PPOConfig:
    horizon: int = 128
    epochs: int = 4
    minibatch: int = 4096
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: float = 0.025


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 7):
        super().__init__()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.actor = self._mlp(obs_dim, action_dim)
        self.critic = self._mlp(obs_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)

    @staticmethod
    def _mlp(inputs, outputs):
        return nn.Sequential(nn.Linear(inputs, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, outputs))

    def distribution(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.clamp(-3, 0.5).exp())

    def act(self, obs, deterministic=False):
        distribution = self.distribution(obs)
        raw = distribution.mean if deterministic else distribution.sample()
        # Environment applies tanh. Ratios can use the raw Gaussian density:
        # the same bijective tanh Jacobian cancels between old/new policies.
        return raw, distribution.log_prob(raw).sum(-1), self.critic(obs).squeeze(-1)


def generalized_advantage(rewards, values, next_values, terminated, truncated, gamma, gae_lambda):
    """Bootstrap timeouts from their FINAL observation, never from the reset state.

    All tensors have shape [time, environments]. Either kind of episode boundary
    stops GAE recursion; only a real termination suppresses value bootstrapping.
    """
    advantages = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        delta = rewards[step] + gamma * next_values[step] * (~terminated[step]) - values[step]
        continuation = ~(terminated[step] | truncated[step])
        last = delta + gamma * gae_lambda * continuation * last
        advantages[step] = last
    return advantages, advantages + values


class PPO:
    def __init__(self, model, cfg: PPOConfig):
        self.model, self.cfg = model, cfg
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

    def update(self, observations, actions, old_log_probs, returns, advantages):
        cfg = self.cfg
        n = len(observations)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        metrics = []
        for _ in range(cfg.epochs):
            order = torch.randperm(n, device=observations.device)
            for ids in order.split(min(cfg.minibatch, n)):
                distribution = self.model.distribution(observations[ids])
                log_prob = distribution.log_prob(actions[ids]).sum(-1)
                log_ratio = log_prob - old_log_probs[ids]
                ratio = log_ratio.exp()
                approx_kl = ((ratio - 1) - log_ratio).mean()
                if approx_kl.item() > 1.5 * cfg.target_kl:
                    return self._metrics(metrics, approx_kl.item())
                surrogate = torch.minimum(ratio * advantages[ids], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantages[ids])
                value = self.model.critic(observations[ids]).squeeze(-1)
                value_loss = (value - returns[ids]).square().mean()
                entropy = distribution.entropy().sum(-1).mean()
                loss = -surrogate.mean() + cfg.value_coef * value_loss - cfg.entropy * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                metrics.append((loss.item(), value_loss.item(), entropy.item(), gradient.item(), approx_kl.item()))
        return self._metrics(metrics)

    @staticmethod
    def _metrics(rows, stopped_kl=None):
        names = ("loss", "value_loss", "entropy", "gradient_norm", "kl")
        result = {key: sum(row[i] for row in rows) / max(1, len(rows)) for i, key in enumerate(names)}
        result["minibatches"] = len(rows)
        if stopped_kl is not None:
            result["early_stop_kl"] = stopped_kl
        return result
