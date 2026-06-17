"""Record trained agents acting, and bundle results, into viewer/data.js.

Produces a single self-contained JS file (`window.RSRL_DATA = {...}`) that the
browser dashboard (viewer/index.html) loads with no server and no network — so
you can open the dashboard by double-clicking the HTML file and watch the
trained agents act.

What it captures:
  * gridworld: two replays sharing the SAME world layout — a full-stack group and
    a survival-only group — so you can flip between them and *see* the difference
    (the stack avoids hazards and clusters; the baseline does not). Each agent
    frame also carries its live drive urgencies + dominant drive, so the
    dashboard can show the steering subsystem switching in real time.
  * tug: one greedy episode of the trained cooperative-control team.
  * results: the committed results/*.json, embedded for the charts.

Run:  python -m viewer.record
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlstack import make_agent  # noqa: E402
from rlstack.agent import BrainAgent  # noqa: E402
from rlstack.drives import ExtrinsicDrive, SocialDrive  # noqa: E402
from rlstack.envs import ResourceWorld, TugGame  # noqa: E402
from rlstack.envs.gridworld import FOOD, HAZARD  # noqa: E402
from rlstack.steering import SteeringSubsystem  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "viewer")
RESULTS = os.path.join(ROOT, "results")


def _food_coords(env: ResourceWorld) -> list[list[int]]:
    ys, xs = np.where(env._grid == FOOD)
    return [[int(x), int(y)] for x, y in zip(xs.tolist(), ys.tolist())]


def _hazard_coords(env: ResourceWorld) -> list[list[int]]:
    ys, xs = np.where(env._grid == HAZARD)
    return [[int(x), int(y)] for x, y in zip(xs.tolist(), ys.tolist())]


def record_grid_variant(drive_names, *, n_agents=5, train_steps=40_000,
                        record_frames=320, seed=7) -> dict:
    env = ResourceWorld(width=18, height=18, n_food=42, n_hazards=12,
                        regrow_delay=14, vision_radius=4, seed=seed)
    agents = [make_agent(i, env.n_actions, drive_names, seed=seed + i) for i in range(n_agents)]
    env.reset(agents)

    # Train (with exploration).
    obs = env.observations()
    for _ in range(train_steps):
        actions = [agents[i].act(obs[i]) for i in range(n_agents)]
        ctxs = env.step(actions)
        rewards = [agents[i].reward_from(ctxs[i]) for i in range(n_agents)]
        nxt = env.observations()
        for i in range(n_agents):
            agents[i].observe(obs[i], rewards[i], nxt[i], done=agents[i].dead)
        env.respawn_dead()
        obs = env.observations()

    # Record greedy behaviour.
    hazards = _hazard_coords(env)
    frames = []
    obs = env.observations()
    for _ in range(record_frames):
        actions = [agents[i].act(obs[i], greedy=True) for i in range(n_agents)]
        ctxs = env.step(actions)
        a_rows = []
        for i in range(n_agents):
            agents[i].reward_from(ctxs[i])  # populate steering.last_info
            info = agents[i].steering.last_info
            urg = info.get("urgencies", {})
            dom = info.get("dominant")
            dom_idx = drive_names.index(dom) if dom in drive_names else 0
            x, y = env.positions[i]
            energy = int(round(agents[i].steering.levels().get("hunger", 1.0) * 100))
            row = [x, y, dom_idx, energy] + [round(float(urg.get(n, 0.0)), 3) for n in drive_names]
            a_rows.append(row)
        frames.append({"a": a_rows, "f": _food_coords(env)})
        env.respawn_dead()
        obs = env.observations()

    return {"drives": list(drive_names), "frames": frames, "hazards": hazards,
            "W": env.W, "H": env.H}


def record_tug(*, train_episodes=6000, seed=3) -> dict:
    env = TugGame(seed=seed)
    team = []
    for i in range(3):
        steering = SteeringSubsystem(
            [ExtrinsicDrive("task", reward_key="extrinsic"),
             SocialDrive("coop", coop_gain=0.4, affiliation_gain=0.0)],
            mode="sum",
        )
        team.append(BrainAgent(i, TugGame.n_actions, steering, alpha=0.2, gamma=0.95,
                               epsilon=0.3, epsilon_min=0.03, epsilon_decay=0.9996, seed=seed + i))

    def episode(train):
        env.reset(team)
        obs = env.observations()
        rec = []
        done = False
        while not done:
            actions = [team[i].act(obs[i], greedy=not train) for i in range(3)]
            ctxs, done = env.step(actions)
            rewards = [team[i].reward_from(ctxs[i]) for i in range(3)]
            nxt = env.observations()
            if train:
                for i in range(3):
                    team[i].observe(obs[i], rewards[i], nxt[i], done)
            else:
                rec.append({
                    "anchors": [[round(float(env.anchors[i, 0]), 2),
                                 round(float(env.anchors[i, 1]), 2)] for i in range(3)],
                    "band": [round(float(env.band_center[0]), 2), round(float(env.band_center[1]), 2)],
                    "bottle": [round(float(env.bottle[0]), 2), round(float(env.bottle[1]), 2)],
                    "gripped": bool(env._gripped()),
                })
            obs = nxt
        return rec

    untrained = episode(train=False)  # greedy on a blank Q-table = uncoordinated flailing
    for _ in range(train_episodes):
        episode(train=True)
    trained = episode(train=False)
    return {"size": env.size, "goal": [float(env.goal[0]), float(env.goal[1])],
            "untrained": untrained, "trained": trained}


def load_results() -> dict:
    out = {}
    for name in ("prisoners", "gridworld", "tug", "evolve"):
        path = os.path.join(RESULTS, f"{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                out[name] = json.load(f)
    return out


def main() -> None:
    print("recording gridworld (stacked)...")
    stacked = record_grid_variant(("hunger", "safety", "social", "curiosity"))
    print("recording gridworld (survival)...")
    survival = record_grid_variant(("hunger",))
    print("recording tug...")
    tug = record_tug()

    data = {
        "gridworld": {
            "W": stacked["W"], "H": stacked["H"], "hazards": stacked["hazards"],
            "variants": {
                "stacked": {"drives": stacked["drives"], "frames": stacked["frames"]},
                "survival": {"drives": survival["drives"], "frames": survival["frames"]},
            },
        },
        "tug": tug,
        "results": load_results(),
    }

    os.makedirs(VIEWER, exist_ok=True)
    out_path = os.path.join(VIEWER, "data.js")
    with open(out_path, "w") as f:
        f.write("// Auto-generated by viewer/record.py — do not edit by hand.\n")
        f.write("window.RSRL_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    size_kb = os.path.getsize(out_path) / 1024
    print(f"wrote {out_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
