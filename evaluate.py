"""Evaluate clean, random-noise, FGSM, and PGD observation attacks."""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from attacks import make_adversarial_observation, predict_action_from_obs


def run_episode(model: DQN, env: gym.Env, attack: str, epsilon: float, pgd_steps: int, objective: str) -> tuple[float, int, int, bool]:
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    changed = 0

    while not done:
        result = make_adversarial_observation(
            model=model,
            env=env,
            obs=obs,
            attack=attack,
            epsilon=epsilon,
            pgd_alpha=max(epsilon / max(pgd_steps, 1), 1e-6),
            pgd_steps=pgd_steps,
            objective=objective,
        )
        obs, reward, terminated, truncated, _ = env.step(result.adv_action)
        done = terminated or truncated
        total_reward += float(reward)
        steps += 1
        changed += int(result.action_changed)
    success = bool(obs[0] >= 0.5)  # MountainCar goal position
    return total_reward, steps, changed, success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MountainCar-v0")
    parser.add_argument("--model-path", default="models/evasionrl_dqn_mountaincar.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--objective", choices=["flip_margin", "reduce_best_q"], default="flip_margin")
    args = parser.parse_args()

    env = gym.make(args.env)
    model = DQN.load(args.model_path, env=env)

    rows = []
    for attack in ["none", "random", "fgsm", "pgd"]:
        rewards, lengths, changes, successes = [], [], [], []
        for _ in range(args.episodes):
            reward, steps, changed, success = run_episode(model, env, attack, args.epsilon, args.pgd_steps, args.objective)
            rewards.append(reward)
            lengths.append(steps)
            changes.append(changed / max(steps, 1))
            successes.append(success)
        rows.append(
            {
                "attack": attack,
                "avg_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "avg_episode_length": np.mean(lengths),
                "success_rate": np.mean(successes),
                "action_change_rate": np.mean(changes),
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
