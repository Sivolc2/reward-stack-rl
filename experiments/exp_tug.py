"""Experiment 3 - TugGame: can independent learners coordinate a shared actuator?

Three agents each control one rope; the bottle only moves when their pulls
triangulate and grip it together. The reward is *shared* — there is no way to
score individually without coordinating. We ask two things:

  1. Do independent Q-learners, each maximising the shared cooperative reward,
     learn to deliver the bottle (cooperation emerging from learning alone)?
  2. Does stacking an explicit cooperation/social drive on top of the shared
     task reward speed that up?

We report delivery rate and mean steps-to-deliver, before vs after training,
for a payoff-only stack and a payoff+cooperation stack.

Run:  python -m experiments.exp_tug
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlstack.agent import BrainAgent  # noqa: E402
from rlstack.drives import ExtrinsicDrive, SocialDrive  # noqa: E402
from rlstack.envs import TugGame  # noqa: E402
from rlstack.steering import SteeringSubsystem  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _make_team(coop_drive: bool, seed: int) -> list[BrainAgent]:
    team = []
    for i in range(3):
        drives = [ExtrinsicDrive("task", reward_key="extrinsic")]
        if coop_drive:
            # Rewards being gripped/coordinated, on top of the shared task signal.
            drives.append(SocialDrive("coop", coop_gain=0.4, affiliation_gain=0.0))
        steering = SteeringSubsystem(drives, mode="sum")
        team.append(BrainAgent(i, TugGame.n_actions, steering,
                               alpha=0.2, gamma=0.95, epsilon=0.3,
                               epsilon_min=0.03, epsilon_decay=0.9996, seed=seed + i))
    return team


def _run_episode(env: TugGame, team, *, train: bool):
    env.reset(team)
    obs = env.observations()
    steps = 0
    while True:
        actions = [team[i].act(obs[i], greedy=not train) for i in range(3)]
        contexts, done = env.step(actions)
        rewards = [team[i].reward_from(contexts[i]) for i in range(3)]
        next_obs = env.observations()
        if train:
            for i in range(3):
                team[i].observe(obs[i], rewards[i], next_obs[i], done)
        obs = next_obs
        steps += 1
        if done:
            break
    delivered = env._dist_to_goal() <= env.grip_radius
    return delivered, steps


def run_condition(coop_drive: bool, *, episodes: int = 6000, seed: int = 0) -> dict:
    env = TugGame(seed=seed)
    team = _make_team(coop_drive, seed)

    def evaluate(n=300):
        d = 0
        s = 0
        for k in range(n):
            delivered, steps = _run_episode(env, team, train=False)
            d += int(delivered)
            s += steps
        return d / n, s / n

    pre_rate, pre_steps = evaluate()
    curve = []
    for ep in range(episodes):
        _run_episode(env, team, train=True)
        if ep % 500 == 0:
            r, _ = evaluate(100)
            curve.append(r)
    post_rate, post_steps = evaluate()
    return {
        "coop_drive": coop_drive,
        "pre_delivery_rate": pre_rate,
        "post_delivery_rate": post_rate,
        "pre_mean_steps": pre_steps,
        "post_mean_steps": post_steps,
        "learning_curve": curve,
    }


def main() -> dict:
    print("=" * 64)
    print("EXPERIMENT 3 - TugGame (cooperative joint-action control)")
    print("=" * 64)
    plain = run_condition(coop_drive=False, seed=3)
    coop = run_condition(coop_drive=True, seed=3)

    print(f"\n{'condition':<26}{'delivery (pre->post)':>22}{'steps (pre->post)':>20}")
    print("-" * 68)
    for label, r in [("task reward only", plain), ("+ cooperation drive", coop)]:
        print(f"{label:<26}"
              f"{r['pre_delivery_rate']:>9.0%} -> {r['post_delivery_rate']:<9.0%}"
              f"{r['pre_mean_steps']:>9.0f} -> {r['post_mean_steps']:<9.0f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "tug.json"), "w") as f:
        json.dump({"plain": plain, "coop": coop}, f, indent=2)
    print("\nsaved -> results/tug.json")
    return {"plain": plain, "coop": coop}


if __name__ == "__main__":
    main()
