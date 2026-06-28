"""Train a DQN moving-car agent for EvasionRL.

This version is tuned to avoid a common MountainCar problem: a weak DQN learns a
boring policy such as always pushing left/right and never reaches the flag.

By default, the agent is trained with reward shaping. The demo/evaluation still
runs in the normal MountainCar-v0 environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import RewardWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


class MountainCarShapedReward(RewardWrapper):
    """Reward shaping for training only.

    Original MountainCar reward is -1 at every step until the car reaches the
    flag. That sparse reward often makes short DQN training runs learn a
    degenerate policy. This shaping gives the agent a small hint for moving to
    the right and building velocity.
    """

    def reward(self, reward: float) -> float:
        position = float(self.env.unwrapped.state[0])
        velocity = float(self.env.unwrapped.state[1])
        shaped = reward + 0.10 * position + 10.0 * abs(velocity)
        if position >= 0.5:
            shaped += 100.0
        return float(shaped)


def make_train_env(env_id: str, shaped: bool) -> gym.Env:
    env = gym.make(env_id)
    if shaped and env_id == "MountainCar-v0":
        env = MountainCarShapedReward(env)
    return Monitor(env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MountainCar-v0")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--model-path", default="models/evasionrl_dqn_mountaincar.zip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shaped-reward", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    use_shaping = not args.no_shaped_reward
    env = make_train_env(args.env, shaped=use_shaping)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=4e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.35,
        exploration_final_eps=0.02,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1,
        seed=args.seed,
    )

    print("Training DQN...")
    print(f"Environment: {args.env}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Reward shaping: {use_shaping}")
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # Evaluate on the true, unshaped environment. This is the environment used by the demo.
    eval_env = Monitor(gym.make(args.env))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    solved_like = bool(mean_reward > -160)  # MountainCar threshold used here as a practical demo check.

    summary = {
        "env": args.env,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "reward_shaping_used_for_training": use_shaping,
        "evaluation_env": "unshaped " + args.env,
        "mean_reward_20_episodes": float(mean_reward),
        "std_reward_20_episodes": float(std_reward),
        "demo_ready": solved_like,
        "note": "For MountainCar, -200 means time-limit failure. A good demo model should usually score much higher than -200.",
    }
    summary_path = model_path.with_suffix(".training_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nEvaluation on normal MountainCar:")
    print(json.dumps(summary, indent=2))
    if not solved_like:
        print("\nWARNING: The model may still be weak. Try more timesteps, e.g. --timesteps 1000000.")


if __name__ == "__main__":
    main()
