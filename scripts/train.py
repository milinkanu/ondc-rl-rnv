"""
Quick training script for ONDCAgentEnv using Stable-Baselines3 PPO.

Usage:
    python scripts/train.py --timesteps 20000 --output models/ppo_ondc.zip
"""
from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from ondc_env import ONDCAgentEnv, EnvConfig


def train(timesteps: int, output: str, seed: int | None = None) -> None:
    config = EnvConfig(max_steps=50, seed=seed)
    env = ONDCAgentEnv(config)

    print("Checking environment...")
    check_env(env, warn=True)

    print(f"Training PPO for {timesteps} timesteps...")
    model = PPO("MlpPolicy", env, verbose=1, seed=seed)
    model.learn(total_timesteps=timesteps)

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    model.save(output)
    print(f"Model saved to {output}")
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--output", default="models/ppo_ondc.zip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.timesteps, args.output, args.seed)


if __name__ == "__main__":
    main()
