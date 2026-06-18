"""Deploy Neural MMO (128 agents) + bridge the reward-stack steering subsystem.

Neural MMO is the multi-agent / economic substrate for the civilization track:
128 agents forage (food/water), fight, gather, and trade on a shared Market.
Each agent gets the FULL drive stack — including `social`, which the single-agent
Craftax demo couldn't exercise.

For every alive agent, each tick, we read its real state from the observation
(food, water, health, gold, nearby players, position), feed it into the actual
`rlstack` drives, and run the same urgency-weighted arbitration. Then we look at
the WHOLE SOCIETY: which drive holds the wheel across the population, how the
population declines as resources run out, and what the economy looks like.

    .venv/bin/python drive_bridge.py
"""
import os
import sys
import time

import numpy as np

# Reuse the numpy-only core repo (this file is at <repo>/integrations/nmmo/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rlstack.drives import (  # noqa: E402
    CuriosityDrive, HomeostaticDrive, SafetyDrive, SocialDrive,
)

import nmmo  # noqa: E402
from nmmo.entity.entity import EntityState  # noqa: E402

COL = EntityState.State.attr_name_to_col
RES_MAX = 100.0  # nmmo player health/food/water base
NAMES = ["hunger", "thirst", "safety", "social", "curiosity"]


def make_stack():
    s = {
        "hunger": HomeostaticDrive("hunger", "ate"),
        "thirst": HomeostaticDrive("thirst", "drank"),
        "safety": SafetyDrive("safety"),
        "social": SocialDrive("social"),
        "curiosity": CuriosityDrive("curiosity"),
    }
    for d in s.values():
        d.reset()
    return s


def homeostat(d, real_level):
    d._prev_deficit = d._deficit()
    d._level = real_level
    r = d._prev_deficit ** 2 - d._deficit() ** 2
    return r, d.urgency({})


def bridge_step(aid, o, stack, prev_health):
    ent = o["Entity"]
    self_rows = ent[ent[:, COL["id"]] == aid]
    if len(self_rows) == 0:
        return None
    me = self_rows[0]
    food = float(me[COL["food"]]) / RES_MAX
    water = float(me[COL["water"]]) / RES_MAX
    health = float(me[COL["health"]])
    gold = float(me[COL["gold"]])
    pos = (int(me[COL["row"]]), int(me[COL["col"]]))
    # nearby players (npc_type == 0), excluding self and empty rows
    players = ent[(ent[:, COL["npc_type"]] == 0) & (ent[:, COL["id"]] != aid)
                  & (ent[:, COL["id"]] != 0)]
    neighbors = len(players)

    rh, wh = homeostat(stack["hunger"], food)
    rt, wt = homeostat(stack["thirst"], water)
    prox = max(0.0, (RES_MAX - health) / RES_MAX)
    stack["safety"]._proximity = prox
    rs = stack["safety"].reward({"hazard_proximity": prox, "got_hurt": health < prev_health})
    ws = stack["safety"].urgency({})
    stack["social"].step({"neighbors": neighbors})
    rso = stack["social"].reward({"neighbors": neighbors})
    wso = stack["social"].urgency({})
    stack["curiosity"].step({"pos": pos})
    rc = stack["curiosity"].reward({})
    wc = stack["curiosity"].urgency({})

    urg = [wh, wt, ws, wso, wc]
    rew = [rh, rt, rs, rso, rc]
    Z = sum(urg) or 1.0
    R = sum(ri * wi for ri, wi in zip(rew, urg)) / Z
    dom = NAMES[int(np.argmax(urg))]
    return dom, R, health, gold, neighbors


def main(ticks: int = 80):
    print("=" * 70)
    print("NEURAL MMO deploy + reward-stack drive bridge  (128 agents, M4 Max CPU)")
    print("=" * 70)
    env = nmmo.Env()
    obs, info = env.reset()
    start_pop = len(obs)
    stacks = {aid: make_stack() for aid in obs}
    prev_health = {aid: RES_MAX for aid in obs}

    pop_curve, dom_totals = [], {n: 0 for n in NAMES}
    rng = np.random.default_rng(0)
    t0 = time.time()

    for tick in range(ticks):
        actions = {aid: env.action_space(aid).sample() for aid in obs}
        obs, rewards, term, trunc, info = env.step(actions)
        alive = len(obs)
        pop_curve.append(alive)
        tick_gold = 0.0
        for aid, o in obs.items():
            if aid not in stacks:
                stacks[aid] = make_stack()
            res = bridge_step(aid, o, stacks[aid], prev_health.get(aid, RES_MAX))
            if res is None:
                continue
            dom, R, health, gold, neighbors = res
            prev_health[aid] = health
            dom_totals[dom] += 1
            tick_gold += gold
        if tick % 20 == 0 or tick == ticks - 1:
            print(f"  tick {tick:>3}: alive={alive:>3}/{start_pop}  "
                  f"total_gold={tick_gold:>6.0f}  ticks/s={(tick+1)/(time.time()-t0):.1f}")

    print(f"\n  population: {start_pop} -> {pop_curve[-1]} agents over {ticks} ticks "
          f"(a society under resource pressure)")
    total = sum(dom_totals.values()) or 1
    print(f"  dominant-drive share across the whole society (the 'configurator',")
    print(f"  now running in {start_pop} parallel brains):")
    for n in NAMES:
        share = dom_totals[n] / total
        bar = "█" * int(round(share * 34))
        print(f"     {n:<10} {bar:<34} {share:5.1%}")
    print(f"\n  wall-clock: {time.time()-t0:.1f}s for {ticks} ticks x {start_pop} agents")
    print(f"  (Market obs is {obs[list(obs)[0]]['Market'].shape if obs else 'n/a'} — the exchange")
    print(f"   a trained trading/acquisition drive would act on; random policy doesn't trade.)")


if __name__ == "__main__":
    main()
