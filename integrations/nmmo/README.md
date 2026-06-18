# nmmo — drive bridge on a 128-agent world (the civilization / economic track)

Deploys **[Neural MMO](https://neuralmmo.github.io) 2.1** (128 agents, foraging +
combat + a Market exchange) and runs the `rlstack` steering subsystem in every
agent at once.

## What it shows (measured, Apple M4 Max, CPU)

```
128 agents · random policy
  tick  0: alive=128/128   ticks/s=33
  tick 79: alive= 34/128   ticks/s=140
  population: 128 → 34 over 80 ticks   (a society under resource pressure)

  dominant drive across the whole society:
     hunger     ██████████████████████  65.3%
     thirst     ███████                 19.7%
     curiosity  ████                    12.9%
     safety     █                        1.9%
     social                              0.2%
  wall-clock: 0.6s for 80 ticks × 128 agents
```

- **Real civilization-scale dynamics on CPU**, ~140 ticks/s for 128 agents — no
  GPU, no XLA-compile wall. NeuralMMO scales by *agent count*, the better CPU path
  than Craftax's JAX batching.
- **The full drive stack, including `social`** (nearby-player count). Each agent's
  dominant drive is read from its real state (`food / water / health / gold /
  neighbours / position`). With a random policy the society starves, so
  hunger/thirst dominate — the arbitration working across 128 parallel brains.
- **Economy exposed:** every agent observes a `Market (384×16)` and carries
  `gold`. Random agents don't trade — that's the hook the economics work builds on.

## Run

```bash
uv venv -p 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python drive_bridge.py
```

## The economy bridge (`economy.py`)

Adds the new **`AcquisitionDrive`** (a wealth-seeking drive keyed on each agent's
`gold`) to the stack and gives agents a scripted trade policy that buys/sells on
the Market via the legal `ActionTargets` masks. Measured:

```
  tick alive listings meanGold maxGold wealthGini  sells  buys
     0   128        0    10.00      10      0.000     37    91
    50    62       62    10.03      13      0.029   1347  4158
   100    26       26    10.27      16      0.076   1873  5708
   149    17       17    10.47      21      0.132   2152  6508
  final: mean gold 10.5 · max 21 · wealth Gini 0.132 · 6508 buys, 2152 sells
  dominant share: hunger 58% · thirst 19% · curiosity 14% · acquisition 6.5% · safety 2%
```

- **The market clears** — 6,508 buys — once the policy acts on *demand*, not just
  supply. (A market with only sellers never transacts; that was the first lesson.)
- **Wealth inequality (Gini) emerges endogenously**, climbing 0.0 → 0.13 over 150
  ticks as some agents accumulate (max gold 10 → 21) — even with near-random
  behaviour. This is exactly the kind of dynamic an economic model wants to study.
- **The acquisition drive participates** (6.5% dominant share): poorer agents
  (below their wealth target) weight it more, richer agents less.
- It uses a small **starting-capital endowment** (`EXCHANGE_BASE_GOLD = 10`); with
  the stock 1 gold there's no purchasing power and the market can't clear.

Run: `.venv/bin/python economy.py`

## Toward the full economic sim

1. ✅ **Acquisition drive + a clearing market + wealth-inequality metric** — done
   (`economy.py`).
2. **Replace the random policy with a learner** (shared neural policy + the reward
   stack as its reward; nmmo ships PufferLib hooks) so trade is strategic —
   forage, survive, accumulate, specialise.
3. **Generational drift** (`rlstack/evolution.py`) over drive parameters across
   episodes — let the society's drive balance (incl. how acquisitive it is) evolve
   under selection, and watch how that changes inequality.
