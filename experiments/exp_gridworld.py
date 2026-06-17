"""Experiment 2 - ResourceWorld drive ablation: does each drive leave a fingerprint?

Rather than only baseline-vs-full, we run an *ablation*: start from a single
hunger drive and add one drive at a time, then the full stack. Each drive should
move its own target behaviour while the others stay put. This directly answers
the design question "which axis do we move to pull out which behaviour?":

  config              expected effect
  hunger              baseline forager
  hunger+curiosity    explores more (visits more distinct cells)
  hunger+safety       far lower hazard exposure
  hunger+social       clusters with other agents
  full stack          all of the above + goal-switching (the steering signature)

Agents never receive a scalar reward from the world; each builds its own reward
from world events via its steering subsystem.

Run:  python -m experiments.exp_gridworld
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlstack import make_agent  # noqa: E402
from rlstack.envs import ResourceWorld  # noqa: E402
from rlstack.metrics import goal_switch_rate  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

WINDOW = 4000         # window for hazard / clustering / goal-switch stats
EXPLORE_WINDOW = 500  # shorter window for exploration: unique cells / steps

CONFIGS = {
    "hunger": ("hunger",),
    "+curiosity": ("hunger", "curiosity"),
    "+safety": ("hunger", "safety"),
    "+social": ("hunger", "social"),
    "full stack": ("hunger", "safety", "social", "curiosity"),
}


def run_group(drive_names, *, n_agents: int = 6, steps: int = 50_000, seed: int = 0) -> dict:
    env = ResourceWorld(width=20, height=20, n_food=55, n_hazards=16,
                        regrow_delay=14, vision_radius=4, seed=seed)
    agents = [make_agent(i, env.n_actions, drive_names, seed=seed + i) for i in range(n_agents)]
    env.reset(agents)

    measure_from = steps - WINDOW
    explore_from = steps - EXPLORE_WINDOW
    explore_cells = [set() for _ in agents]
    hazard_steps = np.zeros(n_agents)
    neighbor_sum = np.zeros(n_agents)
    measured = 0

    obs = env.observations()
    for t in range(steps):
        actions = [agents[i].act(obs[i]) for i in range(n_agents)]
        contexts = env.step(actions)
        rewards = [agents[i].reward_from(contexts[i]) for i in range(n_agents)]
        next_obs = env.observations()
        dead_flags = [agents[i].dead for i in range(n_agents)]
        for i in range(n_agents):
            agents[i].observe(obs[i], rewards[i], next_obs[i], done=dead_flags[i])
        if t >= measure_from:
            for i in range(n_agents):
                if contexts[i]["hazard_proximity"] >= 0.5:
                    hazard_steps[i] += 1
                neighbor_sum[i] += contexts[i]["neighbors"]
            measured += 1
        if t >= explore_from:
            for i in range(n_agents):
                explore_cells[i].add(contexts[i]["pos"])
        env.respawn_dead()
        obs = env.observations()

    per_agent = []
    for i, ag in enumerate(agents):
        warm = len(ag.dominant_history) - measured
        per_agent.append(
            {
                "exploration": len(explore_cells[i]) / EXPLORE_WINDOW,
                "hazard_exposure": hazard_steps[i] / measured,
                "clustering": neighbor_sum[i] / measured,
                "goal_switch_rate": goal_switch_rate(ag.dominant_history[warm:]),
                "deaths": int(env.deaths[i]),
            }
        )

    keys = ["exploration", "hazard_exposure", "clustering", "goal_switch_rate", "deaths"]
    out = {k: float(np.mean([p[k] for p in per_agent])) for k in keys}
    out["sample_map"] = env.render()
    return out


def main() -> dict:
    print("=" * 72)
    print("EXPERIMENT 2 - ResourceWorld drive ablation")
    print("=" * 72)
    results = {name: run_group(drives, seed=7) for name, drives in CONFIGS.items()}

    cols = [
        ("explore", "exploration", "{:.2f}"),
        ("hazard%", "hazard_exposure", "{:.1%}"),
        ("cluster", "clustering", "{:.2f}"),
        ("goalsw%", "goal_switch_rate", "{:.1%}"),
        ("deaths", "deaths", "{:.0f}"),
    ]
    header = f"{'config':<14}" + "".join(f"{c[0]:>10}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for name in CONFIGS:
        r = results[name]
        line = f"{name:<14}" + "".join(f"{fmt.format(r[key]):>10}" for _, key, fmt in cols)
        print(line)
    print("\ndrive -> metric it should move:  curiosity->explore  safety->hazard%(down)"
          "  social->cluster  stack->goalsw%")

    print("\nsample full-stack map (digits=agents, ♣=food, ▲=hazard):")
    print(results["full stack"]["sample_map"])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "gridworld.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved -> results/gridworld.json")
    return results


if __name__ == "__main__":
    main()
