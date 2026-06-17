"""Terminal live viewer — watch the real simulation run, in colour, live.

Unlike the browser dashboard (which replays a recording), this runs the actual
environment step by step in your terminal so you can watch agents act in real
time, with a live read-out of the selected agent's drive urgencies and which
drive currently holds the steering wheel.

Usage:
    python -m viewer.live grid       # ResourceWorld, full drive stack (default)
    python -m viewer.live grid --survival   # the single-reward baseline
    python -m viewer.live tug        # cooperative control game

Press Ctrl-C to quit. Pure stdlib + numpy; no extra dependencies.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlstack import make_agent  # noqa: E402
from rlstack.agent import BrainAgent  # noqa: E402
from rlstack.drives import ExtrinsicDrive, SocialDrive  # noqa: E402
from rlstack.envs import ResourceWorld, TugGame  # noqa: E402
from rlstack.envs.gridworld import FOOD, HAZARD  # noqa: E402
from rlstack.steering import SteeringSubsystem  # noqa: E402

# ANSI helpers
CLS = "\033[2J\033[H"
HIDE, SHOW = "\033[?25l", "\033[?25h"
RESET = "\033[0m"
C = {  # drive -> ansi colour
    "hunger": "\033[33m", "safety": "\033[31m", "social": "\033[34m",
    "curiosity": "\033[32m", "dim": "\033[90m", "white": "\033[97m",
    "food": "\033[32m", "hazard": "\033[91m",
}


def bar(value: float, scale: float = 6.0, width: int = 26) -> str:
    n = max(0, min(width, int(round(value / scale * width))))
    return "█" * n + "·" * (width - n)


def run_grid(survival: bool, train_steps: int = 25_000, fps: float = 8.0) -> None:
    drives = ("hunger",) if survival else ("hunger", "safety", "social", "curiosity")
    env = ResourceWorld(width=18, height=14, n_food=34, n_hazards=10, seed=7)
    agents = [make_agent(i, env.n_actions, drives, seed=7 + i) for i in range(5)]
    env.reset(agents)

    sys.stdout.write(f"training {'survival' if survival else 'full-stack'} agents "
                     f"({train_steps} steps)...\n")
    sys.stdout.flush()
    obs = env.observations()
    for _ in range(train_steps):
        actions = [agents[i].act(obs[i]) for i in range(len(agents))]
        ctxs = env.step(actions)
        rs = [agents[i].reward_from(ctxs[i]) for i in range(len(agents))]
        nxt = env.observations()
        for i in range(len(agents)):
            agents[i].observe(obs[i], rs[i], nxt[i], done=agents[i].dead)
        env.respawn_dead()
        obs = env.observations()

    glyph_food, glyph_haz = C["food"] + "♣" + RESET, C["hazard"] + "▲" + RESET
    sys.stdout.write(HIDE)
    try:
        obs = env.observations()
        t = 0
        while True:
            actions = [agents[i].act(obs[i], greedy=True) for i in range(len(agents))]
            ctxs = env.step(actions)
            for i in range(len(agents)):
                agents[i].reward_from(ctxs[i])  # refresh steering.last_info

            # render map
            rows = [[C["dim"] + "·" + RESET for _ in range(env.W)] for _ in range(env.H)]
            ys, xs = (env._grid == FOOD).nonzero()
            for y, x in zip(ys.tolist(), xs.tolist()):
                rows[y][x] = glyph_food
            ys, xs = (env._grid == HAZARD).nonzero()
            for y, x in zip(ys.tolist(), xs.tolist()):
                rows[y][x] = glyph_haz
            for i, (x, y) in enumerate(env.positions):
                dom = agents[i].steering.last_info.get("dominant", "white")
                rows[y][x] = C.get(dom, C["white"]) + str(i) + RESET

            out = [CLS, C["white"], f"  ResourceWorld — {'SURVIVAL (hunger only)' if survival else 'FULL STACK'}"
                   f"   step {t}   (Ctrl-C to quit)", RESET, ""]
            out += ["  " + "".join(r) for r in rows]
            out.append("")
            # drive panel for agent 0
            a0 = agents[0]
            info = a0.steering.last_info
            urg = info.get("urgencies", {})
            dom = info.get("dominant", "")
            out.append(f"  agent 0 — dominant: {C.get(dom, '')}{dom}{RESET}")
            for name in drives:
                col = C.get(name, "")
                out.append(f"    {col}{name:<10}{RESET} {col}{bar(urg.get(name, 0.0))}{RESET}"
                           f" {urg.get(name, 0.0):.2f}")
            deaths = int(env.deaths.sum())
            out.append("")
            out.append(f"  {C['dim']}total deaths: {deaths}   "
                       f"{'(hunger only — ignores hazards & company)' if survival else '(avoids hazards, clusters, explores)'}{RESET}")
            sys.stdout.write("\n".join(out) + "\n")
            sys.stdout.flush()

            env.respawn_dead()
            obs = env.observations()
            t += 1
            time.sleep(1.0 / fps)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(SHOW + RESET + "\n")


def run_tug(train_episodes: int = 6000, fps: float = 4.0) -> None:
    env = TugGame(seed=3)
    team = []
    for i in range(3):
        steering = SteeringSubsystem(
            [ExtrinsicDrive("task", reward_key="extrinsic"),
             SocialDrive("coop", coop_gain=0.4, affiliation_gain=0.0)], mode="sum")
        team.append(BrainAgent(i, TugGame.n_actions, steering, alpha=0.2, gamma=0.95,
                               epsilon=0.3, epsilon_min=0.03, epsilon_decay=0.9996, seed=3 + i))

    def episode(train):
        env.reset(team)
        obs = env.observations()
        done = False
        while not done:
            acts = [team[i].act(obs[i], greedy=not train) for i in range(3)]
            ctxs, done = env.step(acts)
            rs = [team[i].reward_from(ctxs[i]) for i in range(3)]
            nxt = env.observations()
            if train:
                for i in range(3):
                    team[i].observe(obs[i], rs[i], nxt[i], done)
            obs = nxt
            if not train:
                yield

    print(f"training tug team ({train_episodes} episodes)...")
    for _ in range(train_episodes):
        list(episode(train=True))

    sys.stdout.write(HIDE)
    try:
        while True:
            for _ in episode(train=False):
                size = env.size
                grid = [[C["dim"] + "·" + RESET for _ in range(size)] for _ in range(size)]

                def put(p, ch):
                    x, y = int(round(p[0])), int(round(p[1]))
                    if 0 <= x < size and 0 <= y < size:
                        grid[y][x] = ch
                put(env.goal, C["white"] + "B" + RESET)
                colors = [C["hunger"], C["social"], C["curiosity"]]
                gripped = env._gripped()
                bottle_col = C["curiosity"] if gripped else C["hazard"]
                put(env.bottle, bottle_col + "O" + RESET)
                put(env.band_center, C["white"] + "+" + RESET)
                for i in range(3):
                    put(env.anchors[i], colors[i] + str(i) + RESET)
                dist = ((env.bottle[0] - env.goal[0]) ** 2 + (env.bottle[1] - env.goal[1]) ** 2) ** 0.5
                out = [CLS, C["white"] + "  TugGame — cooperative control (trained)   (Ctrl-C to quit)" + RESET, ""]
                out += ["  " + "".join(r) for r in grid]
                out += ["", f"  gripped: {(C['curiosity']+'yes') if gripped else (C['hazard']+'no')}{RESET}"
                        f"   distance to goal: {dist:.2f}"]
                sys.stdout.write("\n".join(out) + "\n")
                sys.stdout.flush()
                time.sleep(1.0 / fps)
            time.sleep(0.8)  # pause between episodes
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(SHOW + RESET + "\n")


def main() -> None:
    args = sys.argv[1:]
    mode = args[0] if args and not args[0].startswith("-") else "grid"
    survival = "--survival" in args
    if mode == "tug":
        run_tug()
    else:
        run_grid(survival=survival)


if __name__ == "__main__":
    main()
