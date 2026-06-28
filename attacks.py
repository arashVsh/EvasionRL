"""Observation-space evasion attacks for Stable-Baselines3 DQN agents.

Threat model:
- The attacker cannot modify the model, environment dynamics, or reward.
- The attacker only perturbs the observation shown to the trained policy at test time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN

AttackName = Literal["none", "random", "fgsm", "pgd"]
AttackObjective = Literal["reduce_best_q", "flip_margin"]


@dataclass
class AttackResult:
    clean_obs: np.ndarray
    adv_obs: np.ndarray
    perturbation: np.ndarray
    clean_action: int
    adv_action: int
    action_changed: bool
    clean_q_values: np.ndarray
    adv_q_values: np.ndarray
    target_action: int | None = None


def _obs_bounds(env: gym.Env) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(env.observation_space.low, dtype=np.float32)
    high = np.asarray(env.observation_space.high, dtype=np.float32)
    low = np.where(np.isfinite(low), low, -10.0).astype(np.float32)
    high = np.where(np.isfinite(high), high, 10.0).astype(np.float32)
    return low, high


def _q_values(model: DQN, obs_tensor: torch.Tensor) -> torch.Tensor:
    return model.q_net(obs_tensor)


def q_values_np(model: DQN, obs: np.ndarray) -> np.ndarray:
    obs_tensor = torch.tensor(obs.astype(np.float32), dtype=torch.float32, device=model.device).unsqueeze(0)
    with torch.no_grad():
        q = _q_values(model, obs_tensor)
    return q.squeeze(0).detach().cpu().numpy().astype(np.float32)


def predict_action_from_obs(model: DQN, obs: np.ndarray) -> int:
    action, _ = model.predict(obs.astype(np.float32), deterministic=True)
    return int(action)


def _attack_loss(q: torch.Tensor, clean_action: int, target_action: int | None, objective: AttackObjective) -> torch.Tensor:
    """Return a scalar to MINIMIZE by changing the observation.

    reduce_best_q: lower Q(clean greedy action).
    flip_margin: lower Q(clean greedy action) and raise Q(target action), where target is usually
    the best competing action. This tends to make action changes more visible in the dashboard.
    """
    if objective == "reduce_best_q" or target_action is None:
        return q[0, clean_action]
    return q[0, clean_action] - q[0, target_action]


def _best_competing_action(q_np: np.ndarray, clean_action: int) -> int:
    masked = q_np.copy()
    masked[clean_action] = -np.inf
    return int(np.argmax(masked))


def fgsm_attack(
    model: DQN,
    env: gym.Env,
    obs: np.ndarray,
    epsilon: float,
    objective: AttackObjective = "flip_margin",
) -> tuple[np.ndarray, int | None]:
    if epsilon <= 0:
        return obs.astype(np.float32).copy(), None

    low, high = _obs_bounds(env)
    obs_np = obs.astype(np.float32)
    clean_q = q_values_np(model, obs_np)
    clean_action = int(np.argmax(clean_q))
    target_action = _best_competing_action(clean_q, clean_action) if objective == "flip_margin" else None

    obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=model.device).unsqueeze(0)
    obs_tensor.requires_grad_(True)
    q = _q_values(model, obs_tensor)
    loss = _attack_loss(q, clean_action, target_action, objective)

    model.policy.zero_grad()
    loss.backward()
    grad_sign = obs_tensor.grad.detach().sign().squeeze(0).cpu().numpy()

    # Minimize the chosen loss with an L-infinity bounded step.
    adv = obs_np - epsilon * grad_sign
    adv = np.clip(adv, obs_np - epsilon, obs_np + epsilon)
    adv = np.clip(adv, low, high)
    return adv.astype(np.float32), target_action


def pgd_attack(
    model: DQN,
    env: gym.Env,
    obs: np.ndarray,
    epsilon: float,
    alpha: float,
    steps: int,
    objective: AttackObjective = "flip_margin",
) -> tuple[np.ndarray, int | None]:
    if epsilon <= 0 or steps <= 0:
        return obs.astype(np.float32).copy(), None

    low, high = _obs_bounds(env)
    clean = obs.astype(np.float32)
    adv = clean.copy()
    clean_q = q_values_np(model, clean)
    clean_action = int(np.argmax(clean_q))
    target_action = _best_competing_action(clean_q, clean_action) if objective == "flip_margin" else None

    for _ in range(steps):
        adv_t = torch.tensor(adv, dtype=torch.float32, device=model.device).unsqueeze(0)
        adv_t.requires_grad_(True)
        q = _q_values(model, adv_t)
        loss = _attack_loss(q, clean_action, target_action, objective)

        model.policy.zero_grad()
        loss.backward()
        grad_sign = adv_t.grad.detach().sign().squeeze(0).cpu().numpy()

        adv = adv - alpha * grad_sign
        adv = np.clip(adv, clean - epsilon, clean + epsilon)
        adv = np.clip(adv, low, high).astype(np.float32)

    return adv, target_action


def make_adversarial_observation(
    model: DQN,
    env: gym.Env,
    obs: np.ndarray,
    attack: AttackName,
    epsilon: float,
    pgd_alpha: float = 0.005,
    pgd_steps: int = 10,
    objective: AttackObjective = "flip_margin",
) -> AttackResult:
    clean_obs = obs.astype(np.float32).copy()
    clean_q = q_values_np(model, clean_obs)
    clean_action = int(np.argmax(clean_q))
    target_action: int | None = None

    if attack == "none" or epsilon <= 0:
        adv_obs = clean_obs.copy()
    elif attack == "random":
        low, high = _obs_bounds(env)
        noise = np.random.uniform(-epsilon, epsilon, size=clean_obs.shape).astype(np.float32)
        adv_obs = np.clip(clean_obs + noise, low, high).astype(np.float32)
    elif attack == "fgsm":
        adv_obs, target_action = fgsm_attack(model, env, clean_obs, epsilon, objective)
    elif attack == "pgd":
        adv_obs, target_action = pgd_attack(model, env, clean_obs, epsilon, pgd_alpha, pgd_steps, objective)
    else:
        raise ValueError(f"Unknown attack: {attack}")

    adv_q = q_values_np(model, adv_obs)
    adv_action = int(np.argmax(adv_q))
    perturbation = adv_obs - clean_obs
    return AttackResult(
        clean_obs=clean_obs,
        adv_obs=adv_obs,
        perturbation=perturbation,
        clean_action=clean_action,
        adv_action=adv_action,
        action_changed=clean_action != adv_action,
        clean_q_values=clean_q,
        adv_q_values=adv_q,
        target_action=target_action,
    )
