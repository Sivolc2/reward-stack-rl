"""A stylized agent-based model of the post-labor transition.

This is the economic application of the reward-stack idea, built to be useful for
the "#post-eco" framework (see ../posteco/README.md). It is a *mechanism* model,
NOT a forecast: it sweeps an automation level and asks "given automation A, which
institutional regime turns the pothole into a sinkhole, and which keeps humans
aboard the flywheel?" — it cannot say whether or when automation actually rises
(Hanson's timing objection is respected; see the caveats in the README).

The story it encodes (after "Potholes on the Way to Utopia"):

  * Supply solves itself: automation raises *potential* output (goods get cheap).
  * Demand does not: realized output is min(capacity, effective demand). If income
    concentrates, the rich hoard, demand collapses, and capacity sits idle — "too
    much of everything and can't move the inventory."
  * As automation A -> 1, the labor share of output -> 0 and the capital share -> 1
    (NAPCS -> 1): "who eats collapses to who owns capital." Redistribution that
    runs through *wages* (the mid-century settlement) stops working.
  * The reward-stack twist: each agent's marginal propensity to consume / hoard is
    set by its drives (subsistence vs acquisition) *modulated by fear vs vision*
    (p_good). Fear amplifies hoarding -> demand collapse (Moloch); a credible
    shared vision flips cooperative spending/investment into the individually
    rational move. This is the steering subsystem at civilizational scale.

Institutional levers (the three pillars + the status quo) live in :class:`Regime`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlstack.metrics import gini  # noqa: E402  (reuse the same inequality metric)


@dataclass
class Regime:
    """An institutional configuration — the simulation's policy levers.

    Pillar 1 (Charter): ``charter_epsilon`` of *capital income* is held in a
    non-dilutable commons pool and paid out per-capita. Because it sits at the
    capital/formation layer it is **not gameable** (``charter_leak = 0``).

    Status-quo redistribution (Piketty-style): ``tax_rate`` on income, but capital
    income evades a fraction ``capital_evasion`` and a fraction ``tax_leak`` is lost
    to routing/admin — "every standard intervention surface is gameable."

    Pillar 2 (Advocates): ``advocate_coverage`` of agents get an advocate that
    *expands* their realized demand (``advocate_mpc_boost``) — surfacing latent
    high-income-elasticity wants so income gets *spent* rather than hoarded ("demand
    must find new places to be put"). This keeps the flywheel turning.

    Pillar 3 (Vision): ``vision_floor`` is the exogenous shared-destination signal
    that sets a floor on p_good (the expectation the future goes well "for me"),
    flipping agents from defensive hoarding to cooperative investment.
    """

    name: str = "laissez-faire"
    tax_rate: float = 0.0
    capital_evasion: float = 0.8      # share of capital income that routes around tax
    tax_leak: float = 0.15            # admin / routing loss on what's collected
    charter_epsilon: float = 0.0      # non-dilutable commons share of capital income
    advocate_coverage: float = 0.0
    advocate_mpc_boost: float = 0.0   # demand expansion for agents with an advocate
    vision_floor: float = 0.0         # exogenous Schelling-point signal in [0,1]


# Preset regimes used by the experiments.
REGIMES = {
    "laissez-faire": Regime("laissez-faire"),
    "wealth-tax": Regime("wealth-tax", tax_rate=0.4),
    "charter": Regime("charter", charter_epsilon=0.20),
    "charter+advocates": Regime("charter+advocates", charter_epsilon=0.20,
                                advocate_coverage=1.0, advocate_mpc_boost=0.4),
    "all-three": Regime("all-three", charter_epsilon=0.20, advocate_coverage=1.0,
                        advocate_mpc_boost=0.4, vision_floor=0.85),
}


class PostLaborEconomy:
    def __init__(
        self,
        automation: float,
        regime: Regime,
        *,
        n_agents: int = 400,
        periods: int = 120,
        needs: float = 1.0,
        base_output: float = 2.5,      # per-capita potential output at A=0 (>needs)
        auto_abundance: float = 4.0,   # extra per-capita potential output at A=1
        invest_return: float = 0.02,   # r: return on capital per period
        fix_belief: float | None = None,  # pin p_good (for fear-vs-vision comparative statics)
        seed: int = 0,
    ) -> None:
        self.A = float(automation)
        self.regime = regime
        self.fix_belief = fix_belief
        self.N = n_agents
        self.T = periods
        self.needs = needs
        self.base_output = base_output
        self.auto_abundance = auto_abundance
        self.invest_return = invest_return
        self.rng = np.random.default_rng(seed)

        # Initial conditions: a buffer of wealth, Pareto-concentrated capital
        # ownership (the starting concentration the transition amplifies or contains).
        self.w = self.rng.lognormal(mean=1.0, sigma=0.4, size=self.N)
        self.k = (self.rng.pareto(1.6, size=self.N) + 0.1)
        self.k *= self.N / self.k.sum()          # normalise capital shares
        self.p_good = fix_belief if fix_belief is not None else max(0.5, regime.vision_floor)
        self.demand = None                        # bootstrapped on first step
        self.history: list[dict] = []

    # --- one period ---------------------------------------------------------
    def step(self) -> dict:
        A, r = self.A, self.regime
        K = self.k.sum()

        # Potential output (supply): automation makes goods abundant/cheap.
        capacity = self.N * (self.base_output + self.auto_abundance * A)
        # Realized output is demand-constrained (the China/Keynes wall).
        if self.demand is None:
            self.demand = capacity
        Y = min(capacity, self.demand)

        # Functional income distribution: labor share (1-A), capital share A.
        labor_total = (1.0 - A) * Y
        capital_total = A * Y
        labor_income = np.full(self.N, labor_total / self.N)  # wages per worker

        # Charter: epsilon of capital income -> non-dilutable per-capita dividend.
        commons = r.charter_epsilon * capital_total
        commons_dividend = np.full(self.N, commons / self.N)
        private_capital_total = capital_total - commons
        capital_income = private_capital_total * (self.k / K)

        gross = labor_income + capital_income + commons_dividend

        # Status-quo tax: capital income largely evades it (routes around), and a
        # slice of what's nominally owed is dodged (tax_leak). The dodged money is
        # NOT destroyed — it stays with the payer; the tax is just less effective.
        # That leakiness is the whole point: "every standard intervention surface is
        # gameable" — only the formation-layer charter is not.
        if r.tax_rate > 0:
            taxable = labor_income + (1.0 - r.capital_evasion) * capital_income
            moved = r.tax_rate * (1.0 - r.tax_leak) * taxable   # what actually redistributes
            net = gross - moved + moved.sum() / self.N
        else:
            net = gross

        available = self.w + net

        # --- the reward-stack at the wallet: consume vs hoard vs invest -------
        # Subsistence drive: cover needs first. Acquisition drive: save the
        # surplus (the rich save a larger share). Fear/vision (p_good) modulates
        # both how much is spent and how much of what's saved is *invested*
        # (productive demand) vs *hoarded* (idle): low p_good -> precautionary
        # hoarding -> demand collapses. This is Moloch at the wallet.
        # Marginal propensity to consume: poor spend nearly all, rich save. Fear
        # (low p_good) suppresses spending across the board (precautionary saving).
        mpc_base = np.clip(1.2 - self.w / (4.0 * max(self.w.mean(), 1e-6)), 0.2, 0.95)
        mpc = mpc_base * (0.5 + 0.5 * self.p_good)
        # Advocates expand demand: they surface latent wants so income gets spent.
        has_advocate = self.rng.random(self.N) < r.advocate_coverage
        mpc = np.where(has_advocate, mpc + r.advocate_mpc_boost * (1.0 - mpc), mpc)

        surplus = np.maximum(available - self.needs, 0.0)
        consumption = np.minimum(available, np.minimum(available, self.needs) + mpc * surplus)
        subsistence_fail = consumption < self.needs - 1e-9

        savings = available - consumption                # >= 0
        invest_frac = 0.2 + 0.7 * self.p_good            # invest vs idle-hoard
        invested = savings * invest_frac                 # productive: counts as demand
        self.w = savings - invested                      # hoarded cash sits idle (no demand)

        # Capital ownership concentrates via investment and returns (r > g), unless
        # a broad dividend (the charter) lets everyone accumulate too.
        self.k = (self.k + invested * 0.05) * (1.0 + self.invest_return)
        self.k *= self.N / self.k.sum()                  # renormalise shares

        # Circular flow (money conserved): next demand = what actually gets spent
        # this period = consumption + investment. Only *hoarded* savings leak out of
        # circulation — that is the underconsumption channel (Keynes/Kumhof-Rancière).
        spending = float(consumption.sum() + invested.sum())
        self.demand = 0.5 * self.demand + 0.5 * spending

        # --- vision / fear contagion -----------------------------------------
        fail_rate = float(subsistence_fail.mean())
        # Belief tracks broad welfare, floored by the exogenous vision signal:
        # widespread subsistence failure breeds fear; a credible vision holds it up.
        if self.fix_belief is not None:
            self.p_good = self.fix_belief                # pinned: comparative statics
        else:
            target = max(r.vision_floor, 1.0 - 2.5 * fail_rate)
            self.p_good += 0.15 * (target - self.p_good)
            self.p_good = float(np.clip(self.p_good, 0.0, 1.0))

        net_worth = self.w + self.k                      # liquid + capital ownership
        rec = {
            "gini": gini(net_worth),
            "top1_capital": float(np.sort(self.k)[-max(1, self.N // 100):].sum() / self.k.sum()),
            "subsistence_fail": fail_rate,
            "output": Y,
            "capacity": capacity,
            "demand_gap": (capacity - Y) / capacity,
            "p_good": self.p_good,
            "median_consumption": float(np.median(consumption)),
        }
        self.history.append(rec)
        return rec

    def run(self) -> dict:
        for _ in range(self.T):
            self.step()
        # Report the average of the last 20% of periods (steady-ish state).
        tail = self.history[-max(1, self.T // 5):]
        keys = tail[0].keys()
        return {k: float(np.mean([h[k] for h in tail])) for k in keys}
