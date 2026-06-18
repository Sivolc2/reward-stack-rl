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

## Toward the civilization / economic sim

1. **Trade / acquisition drive** keyed on `gold` + `Market`, with agents actually
   buying/selling — the seed of an economy. *(in progress)*
2. **Replace random policy** with a learner (shared neural policy + the reward
   stack as its reward; nmmo ships PufferLib hooks).
3. **Generational drift** (`rlstack/evolution.py`) over drive parameters across
   episodes — let the society's drive balance evolve under selection.
