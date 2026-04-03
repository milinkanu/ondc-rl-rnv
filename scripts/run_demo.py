"""
Inference / demo script for ONDCAgentEnv.

Usage:
    python scripts/run_demo.py --model path/to/model.zip [options]

Options:
    --model         Path to a saved Stable-Baselines3 model (.zip)
    --budget        Episode budget (default: 1000.0)
    --urgency       Episode urgency 0.0-1.0 (default: 0.5)
    --target-item   Item name to search for (default: "laptop")
    --max-steps     Maximum steps per episode (default: 50)
    --no-render     Disable per-step rendering
    --seed          Random seed for reproducibility
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from ondc_env import ONDCAgentEnv, EnvConfig
from ondc_env.types import EpisodeState


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    total_reward: float
    steps_taken: int
    terminated: bool
    truncated: bool
    final_phase: str
    total_spent: float
    invalid_action_count: int
    order_status: str | None
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def run_demo(
    model_path: str,
    task_config: dict | None = None,
    render: bool = True,
    max_steps: int = 50,
) -> EpisodeResult:
    """
    Load a trained SB3 model and run a full demo episode.

    Args:
        model_path:  Path to a Stable-Baselines3 .zip model file.
        task_config: Optional dict with keys: budget, urgency, target_item, seed.
        render:      If True, call env.render() after each step.
        max_steps:   Maximum number of steps before truncation.

    Returns:
        EpisodeResult with episode summary.
    """
    # Lazy import so the script still works without SB3 installed for unit tests
    try:
        from stable_baselines3 import PPO  # noqa: F401 — used via load()
        from stable_baselines3.common.base_class import BaseAlgorithm
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required to run the inference script. "
            "Install it with: pip install stable-baselines3"
        ) from exc

    task_config = task_config or {}
    seed: int | None = task_config.get("seed", None)

    env_config = EnvConfig(max_steps=max_steps, seed=seed)
    env = ONDCAgentEnv(env_config)

    # Load model — SB3 models expose a .load() classmethod
    model: BaseAlgorithm = PPO.load(model_path, env=env)

    reset_options: dict[str, Any] = {}
    if "budget" in task_config:
        reset_options["budget"] = float(task_config["budget"])
    if "urgency" in task_config:
        reset_options["urgency"] = float(task_config["urgency"])
    if "target_item" in task_config:
        reset_options["target_item"] = str(task_config["target_item"])

    obs, info = env.reset(seed=seed, options=reset_options or None)

    total_reward = 0.0
    steps_taken = 0
    terminated = False
    truncated = False
    last_info: dict[str, Any] = info
    all_events: list[dict[str, Any]] = []

    if render:
        env.render()

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(int(action))

        total_reward += float(reward)
        steps_taken += 1
        last_info = step_info

        # Collect events
        for evt in step_info.get("events", []):
            all_events.append(
                {"seller_id": evt.seller_id, "event_type": evt.event_type}
                if hasattr(evt, "seller_id")
                else evt
            )

        if render:
            env.render()

        # Safety guard — env should handle this, but be explicit
        if steps_taken >= max_steps:
            truncated = True
            break

    state: EpisodeState = env.state  # type: ignore[attr-defined]

    env.close()

    return EpisodeResult(
        total_reward=total_reward,
        steps_taken=steps_taken,
        terminated=terminated,
        truncated=truncated,
        final_phase=state.current_phase.name,
        total_spent=state.total_spent,
        invalid_action_count=state.invalid_action_count,
        order_status=state.order_status.name if state.order_status is not None else None,
        reward_breakdown=last_info.get("reward_breakdown", {}),
        events=all_events,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a demo episode with a trained ONDCAgentEnv model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to SB3 model .zip file")
    parser.add_argument("--budget", type=float, default=1000.0, help="Episode budget")
    parser.add_argument("--urgency", type=float, default=0.5, help="Urgency level (0.0–1.0)")
    parser.add_argument("--target-item", default="laptop", help="Item to search for")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--no-render", action="store_true", help="Disable per-step rendering")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    task_config = {
        "budget": args.budget,
        "urgency": args.urgency,
        "target_item": args.target_item,
        "seed": args.seed,
    }

    result = run_demo(
        model_path=args.model,
        task_config=task_config,
        render=not args.no_render,
        max_steps=args.max_steps,
    )

    print("\n=== Episode Result ===")
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
