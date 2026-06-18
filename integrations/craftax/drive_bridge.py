"""Deploy Craftax + bridge the reward-stack steering subsystem onto it.

Two things:
  1. RAW THROUGHPUT — jit + lax.scan a random policy to show how fast Craftax
     runs on this laptop (JAX, CPU).
  2. DRIVE BRIDGE — map Craftax's *real* physiological state (food, drink,
     energy, health, position) into our drives, and run the SAME urgency-weighted
     arbitration from rlstack/steering.py each step. This shows the "configurator"
     (the steering subsystem) reacting to a real environment: as food/drink
     deplete, hunger/thirst take the wheel; when health drops, safety does.

Craftax is single-agent, so there is no `social` drive here (that one needs a
multi-agent world — NeuralMMO is the next step). Run:

    .venv/bin/python drive_bridge.py
"""
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

# Reuse the ACTUAL drives + steering math from the numpy-only core repo
# (this file lives at <repo>/integrations/craftax/, so go up three levels).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rlstack.drives import CuriosityDrive, HomeostaticDrive, SafetyDrive  # noqa: E402

from craftax.craftax_env import make_craftax_env_from_name  # noqa: E402

env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=True)
params = env.default_params
N_ACTIONS = env.action_space(params).n


# ---------------------------------------------------------------------------
# 1. Raw throughput: how fast does Craftax step on this machine?
# ---------------------------------------------------------------------------
def raw_throughput(steps: int = 4000) -> None:
    def rollout(rng):
        _, state = env.reset(rng, params)

        def body(carry, _):
            rng, state = carry
            rng, ka, ks = jax.random.split(rng, 3)
            a = jax.random.randint(ka, (), 0, N_ACTIONS)
            obs, state, r, done, info = env.step(ks, state, a, params)
            return (rng, state), r

        (rng, state), rs = jax.lax.scan(body, (rng, state), None, length=steps)
        return rs.sum()

    fast = jax.jit(rollout)
    fast(jax.random.PRNGKey(0)).block_until_ready()  # warm up / compile
    t0 = time.time()
    total = fast(jax.random.PRNGKey(1)).block_until_ready()
    dt = time.time() - t0
    print(f"  raw env throughput: {steps / dt:,.0f} steps/sec  "
          f"({steps} steps in {dt*1000:.0f} ms, single CPU rollout)")


# ---------------------------------------------------------------------------
# 2. Drive bridge: run our steering subsystem on Craftax's real state.
# ---------------------------------------------------------------------------
DRIVE_NAMES = ["hunger", "thirst", "fatigue", "safety", "curiosity"]


def homeostat(d: HomeostaticDrive, real_level: float):
    """Mirror a Craftax physiological variable into a homeostatic drive and
    return (drive-reduction reward, urgency) using the real rlstack formulas."""
    d._prev_deficit = d._deficit()
    d._level = real_level
    r = d._prev_deficit ** 2 - d._deficit() ** 2
    return r, d.urgency({})


def drive_bridge(steps: int = 2000) -> None:
    hunger = HomeostaticDrive("hunger", "ate")
    thirst = HomeostaticDrive("thirst", "drank")
    fatigue = HomeostaticDrive("fatigue", "rested")
    safety = SafetyDrive("safety")
    curiosity = CuriosityDrive("curiosity")
    for d in (hunger, thirst, fatigue, safety, curiosity):
        d.reset()

    rng = jax.random.PRNGKey(2)
    obs, state = env.reset(rng, params)
    prev_health = float(state.player_health)

    dom_counts = {n: 0 for n in DRIVE_NAMES}
    switches = 0
    last_dom = None
    samples = []
    max_achievements = 0

    for step in range(steps):
        rng, ka, ks = jax.random.split(rng, 3)
        a = jax.random.randint(ka, (), 0, N_ACTIONS)
        obs, state, reward, done, info = env.step(ks, state, a, params)

        health = float(state.player_health)
        food = float(state.player_food)
        drink = float(state.player_drink)
        energy = float(state.player_energy)
        pos = (int(state.player_position[0]), int(state.player_position[1]))

        # Bridge real physiology -> drives -> the SAME arbitration as the core repo.
        rh, wh = homeostat(hunger, food / 9.0)
        rt, wt = homeostat(thirst, drink / 9.0)
        rf, wf = homeostat(fatigue, energy / 9.0)
        prox = max(0.0, (9.0 - health) / 9.0)
        safety._proximity = prox
        rs = safety.reward({"hazard_proximity": prox, "got_hurt": health < prev_health})
        ws = safety.urgency({})
        curiosity.step({"pos": pos})
        rc = curiosity.reward({})
        wc = curiosity.urgency({})
        prev_health = health

        rewards = [rh, rt, rf, rs, rc]
        urgencies = [wh, wt, wf, ws, wc]
        Z = sum(urgencies) or 1.0
        R = sum(ri * wi for ri, wi in zip(rewards, urgencies)) / Z   # dynamic arbitration
        dom = DRIVE_NAMES[int(np.argmax(urgencies))]
        dom_counts[dom] += 1
        if last_dom is not None and dom != last_dom:
            switches += 1
        last_dom = dom

        max_achievements = max(max_achievements, int(jnp.sum(state.achievements)))
        if step < 6 or step % 400 == 0:
            samples.append((step, food, drink, energy, health, dom, R))

    print(f"  bridged {steps} steps. achievements unlocked (random policy): {max_achievements}/22")
    print(f"  dominant-drive share over the run (the 'configurator' at work):")
    for n in DRIVE_NAMES:
        share = dom_counts[n] / steps
        bar = "█" * int(round(share * 30))
        print(f"     {n:<10} {bar:<30} {share:5.1%}")
    print(f"  goal-switch rate (dominant drive changed): {switches / (steps-1):.1%}")
    print(f"  sample steps (food/drink/energy/health -> dominant drive, intrinsic R):")
    print(f"     {'step':>5} {'food':>5} {'drink':>5} {'energy':>6} {'health':>6}  {'dominant':<10} {'R':>7}")
    for s, food, drink, energy, health, dom, R in samples:
        print(f"     {s:>5} {food:>5.1f} {drink:>5.1f} {energy:>6.1f} {health:>6.1f}  {dom:<10} {R:>7.3f}")


if __name__ == "__main__":
    print("=" * 68)
    print("CRAFTAX deploy + reward-stack drive bridge  (Apple M4 Max, JAX CPU)")
    print("=" * 68)
    print("\n[1] raw throughput")
    raw_throughput()
    print("\n[2] steering subsystem bridged onto Craftax physiology")
    drive_bridge()
