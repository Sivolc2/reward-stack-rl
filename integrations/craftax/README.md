# craftax — drive bridge on a single-agent survival world

Deploys **[Craftax](https://github.com/MichaelTMatthews/Craftax)** (Crafter +
NetHack, in JAX) and bridges the `rlstack` steering subsystem onto its real
physiology.

## What it shows (measured, Apple M4 Max, JAX CPU)

```
[1] raw throughput: 1,271 steps/sec   (single sequential rollout)

[2] steering subsystem bridged onto Craftax's REAL physiology
    dominant-drive share:  curiosity 45% · thirst 38% · hunger 12% · safety 4%
     step  food drink energy health  dominant
        0   9.0   9.0    9.0    9.0   curiosity   ← all full → explore
      800   3.0   1.0    9.0    7.0   thirst        ← drink critical → thirst
     1200   5.0   6.0    8.0    9.0   hunger        ← food low → hunger
```

The dominant drive isn't scripted — it falls out of which physiological variable
is most depleted (`player_food / drink / energy / health`), using the exact
`rlstack` arbitration. `social` is absent (Craftax is single-agent — see the
`nmmo` integration for that).

## Throughput note (honest)

The "100k+ steps/s" Craftax headlines are **GPU** figures. On **CPU**,
`jax.vmap`-batched rollouts of the symbolic env do *not* pay off (XLA-CPU doesn't
parallelise the heavy per-step logic across cores, and the vmap compile runs into
minutes). Single-env CPU (~1.3k steps/s) is the realistic number here, plenty for
the bridge demo. Real throughput wants a GPU/TPU (or experimental `jax-metal`).

## Run

```bash
uv venv -p 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python drive_bridge.py
```

## Next

Close the world-model gap: add a small learned predictor + planner so the agent
*acts* on the drive signal instead of a random policy (the JEPA/configurator
loop). For multi-agent + economy, see [`../nmmo`](../nmmo).
