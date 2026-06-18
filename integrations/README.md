# integrations/ — the drive stack on real RL environments

The core `rlstack` package is deliberately **numpy-only** so it runs anywhere.
These integrations bridge the *same* drives + steering subsystem onto heavier,
real-world RL environments. They are **optional** and each carries its own
dependencies and its own virtualenv — installing them never touches the core.

The bridge pattern is identical in both: read an environment's real per-agent
state, feed it into the actual `rlstack` drives, and run the same
urgency-weighted arbitration. The dominant drive then falls out of the
environment's dynamics rather than being scripted — the LeCun-"configurator"
role, demonstrated on a real world model.

| Integration | Env | Agents | Why it's here |
|---|---|---|---|
| [`craftax/`](craftax/) | [Craftax](https://github.com/MichaelTMatthews/Craftax) (Crafter+NetHack, JAX) | 1 | proves the bridge on a rich single-agent survival world; watch dominant drive switch curiosity → thirst → hunger as physiology depletes |
| [`nmmo/`](nmmo/) | [Neural MMO](https://neuralmmo.github.io) 2.1 | 128 | the multi-agent / **economic** substrate for the civilization track; exercises the `social` drive and exposes a Market exchange |

## Why two separate venvs

System Python here is 3.14, which predates most ML wheels. Each integration pins
a compatible interpreter:

- `craftax/` → Python **3.12** + JAX (CPU). Single-env ~1.3k steps/s on Apple
  Silicon; batched throughput needs a GPU (see its README).
- `nmmo/` → Python **3.11**. ~140 ticks/s for 128 agents on CPU — scales by agent
  count, the better CPU path.

```bash
cd integrations/<name>
uv venv -p <ver> .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python drive_bridge.py
```

Each `drive_bridge.py` imports `rlstack` from the repo root via a relative path,
so it works wherever the repo is checked out. The `.venv/` directories are
gitignored.
