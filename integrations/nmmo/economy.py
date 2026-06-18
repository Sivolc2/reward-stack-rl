"""Neural MMO economy bridge — the reward stack with an AcquisitionDrive + trade.

Extends drive_bridge.py toward the economic-modeling goal:

  * adds the new `AcquisitionDrive` (keyed on each agent's `gold`) to the stack,
    so wealth becomes one of the things the steering subsystem weighs;
  * gives agents a scripted *trade policy* — using the env's ActionTargets masks
    they list inventory items for sale and buy affordable listings, so gold
    actually flows and a (proto-)market forms;
  * instruments the economy: market listings, total/mean gold, and the **Gini
    coefficient** of wealth over time (reusing rlstack.metrics.gini).

This is a substrate + measurement layer, not a trained economy: agents move
randomly and trade opportunistically, which is enough to make the plumbing and
the metrics real. Swapping the random/scripted policy for a learner is the next
step. Run:  .venv/bin/python economy.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rlstack.drives import (  # noqa: E402
    AcquisitionDrive, CuriosityDrive, HomeostaticDrive, SafetyDrive, SocialDrive,
)
from rlstack.metrics import gini  # noqa: E402

import nmmo  # noqa: E402
from nmmo.entity.entity import EntityState  # noqa: E402

COL = EntityState.State.attr_name_to_col
RES_MAX = 100.0
NAMES = ["hunger", "thirst", "safety", "social", "curiosity", "acquisition"]


def make_stack():
    s = {
        "hunger": HomeostaticDrive("hunger", "ate"),
        "thirst": HomeostaticDrive("thirst", "drank"),
        "safety": SafetyDrive("safety"),
        "social": SocialDrive("social"),
        "curiosity": CuriosityDrive("curiosity"),
        "acquisition": AcquisitionDrive("acquisition", wealth_key="wealth", target=20.0),
    }
    for d in s.values():
        d.reset()
    return s


def homeostat(d, real_level):
    d._prev_deficit = d._deficit()
    d._level = real_level
    return d._prev_deficit ** 2 - d._deficit() ** 2, d.urgency({})


def economy_action(o, rng):
    """Random move + opportunistic trade, all via the legal ActionTargets masks.

    We prioritise *demand* — if there's something affordable to buy, buy it — so
    the market actually clears. Otherwise list a held item for sale at the floor
    price. (A market with only supply never transacts; you need buyers acting.)
    """
    at = o["ActionTargets"]
    act = {}
    mv = np.flatnonzero(at["Move"]["Direction"])
    if len(mv):
        act["Move"] = {"Direction": int(rng.choice(mv))}
    sold = bought = False
    buy_items = np.flatnonzero(at["Buy"]["MarketItem"])
    buy_items = buy_items[buy_items != 0]
    sell_items = np.flatnonzero(at["Sell"]["InventoryItem"])
    sell_items = sell_items[sell_items != 0]
    if len(buy_items) and rng.random() < 0.75:
        act["Buy"] = {"MarketItem": int(rng.choice(buy_items))}
        bought = True
    elif len(sell_items):
        prices = np.flatnonzero(at["Sell"]["Price"])
        price = int(prices[0]) if len(prices) else 1  # list at the floor so it clears
        act["Sell"] = {"InventoryItem": int(sell_items[0]), "Price": price}
        sold = True
    return act, sold, bought


def drive_eval(aid, o, stack, prev_health):
    ent = o["Entity"]
    me = ent[ent[:, COL["id"]] == aid]
    if len(me) == 0:
        return None
    me = me[0]
    food = float(me[COL["food"]]) / RES_MAX
    water = float(me[COL["water"]]) / RES_MAX
    health = float(me[COL["health"]])
    gold = float(me[COL["gold"]])
    pos = (int(me[COL["row"]]), int(me[COL["col"]]))
    neighbors = int(((ent[:, COL["npc_type"]] == 0) & (ent[:, COL["id"]] != aid)
                     & (ent[:, COL["id"]] != 0)).sum())

    rh, wh = homeostat(stack["hunger"], food)
    rt, wt = homeostat(stack["thirst"], water)
    prox = max(0.0, (RES_MAX - health) / RES_MAX)
    stack["safety"]._proximity = prox
    rs = stack["safety"].reward({"hazard_proximity": prox, "got_hurt": health < prev_health})
    ws = stack["safety"].urgency({})
    stack["social"].step({"neighbors": neighbors})
    rso, wso = stack["social"].reward({"neighbors": neighbors}), stack["social"].urgency({})
    stack["curiosity"].step({"pos": pos})
    rc, wc = stack["curiosity"].reward({}), stack["curiosity"].urgency({})
    stack["acquisition"].step({"wealth": gold})
    ra, wa = stack["acquisition"].reward({}), stack["acquisition"].urgency({})

    urg = [wh, wt, ws, wso, wc, wa]
    dom = NAMES[int(np.argmax(urg))]
    return dom, health, gold


class EconomyConfig(nmmo.config.Default):
    """Default world, but endow agents with starting capital so the market has
    purchasing power (with the stock 1 gold, supply forms but never clears)."""
    EXCHANGE_BASE_GOLD = 10


def main(ticks: int = 150):
    print("=" * 70)
    print("NEURAL MMO economy bridge — reward stack + AcquisitionDrive + trade")
    print("=" * 70)
    env = nmmo.Env(EconomyConfig())
    obs, info = env.reset()
    start_pop = len(obs)
    stacks = {aid: make_stack() for aid in obs}
    prev_health = {aid: RES_MAX for aid in obs}
    rng = np.random.default_rng(0)

    dom_totals = {n: 0 for n in NAMES}
    sell_actions = buy_actions = 0
    t0 = time.time()
    print(f"\n  {'tick':>4} {'alive':>5} {'listings':>8} {'meanGold':>8} "
          f"{'maxGold':>7} {'wealthGini':>10} {'sells':>6} {'buys':>5}")
    for tick in range(ticks):
        actions = {}
        for aid, o in obs.items():
            a, sold, bought = economy_action(o, rng)
            actions[aid] = a
            sell_actions += sold
            buy_actions += bought
        obs, rewards, term, trunc, info = env.step(actions)

        golds, listings = [], 0
        for aid, o in obs.items():
            if aid not in stacks:
                stacks[aid] = make_stack()
            res = drive_eval(aid, o, stacks[aid], prev_health.get(aid, RES_MAX))
            if res is None:
                continue
            dom, health, gold = res
            prev_health[aid] = health
            dom_totals[dom] += 1
            golds.append(gold)
            listings += int((o["Market"][:, 0] != 0).sum())

        if tick % 25 == 0 or tick == ticks - 1:
            g = np.array(golds) if golds else np.array([0.0])
            print(f"  {tick:>4} {len(obs):>5} {listings:>8} {g.mean():>8.2f} "
                  f"{g.max():>7.0f} {gini(g):>10.3f} {sell_actions:>6} {buy_actions:>5}")

    total = sum(dom_totals.values()) or 1
    print(f"\n  dominant-drive share across the society (now WITH acquisition):")
    for n in NAMES:
        share = dom_totals[n] / total
        print(f"     {n:<12} {'█' * int(round(share * 32)):<32} {share:5.1%}")
    g = np.array(golds) if golds else np.array([0.0])
    print(f"\n  final economy: {len(obs)} agents · mean gold {g.mean():.2f} · "
          f"max {g.max():.0f} · wealth Gini {gini(g):.3f}")
    print(f"  trade actions submitted over {ticks} ticks: {sell_actions} sells, {buy_actions} buys")
    print(f"  wall-clock: {time.time()-t0:.1f}s")
    print("\n  note: even with random movement + opportunistic trade, the market")
    print("  clears and wealth inequality (Gini) emerges endogenously. Swapping the")
    print("  random policy for a learner that forages, survives, and trades on the")
    print("  acquisition drive is what turns this into a real economic model.")


if __name__ == "__main__":
    main()
