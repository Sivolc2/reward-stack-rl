---
title: "RL Environments & Games Survey — Reward-Stack Testbed"
date: 2026-06-17
tags: [reinforcement-learning, multi-agent, reward-stack, cooperation, social-dilemma, survey, marl]
---

# RL Environments & Games Survey

A landscape review of open-source RL environments for the **reward-stack** testbed — a
multi-agent project testing whether a *stack of many small reward functions* (modeling the
brain's innate "steering subsystem" drives, per [Steven Byrnes / Astera](https://www.alignmentforum.org/posts/hE56gYi5d68uux9oM/intro-to-brain-like-agi-safety-3-two-subsystems-learning-and))
produces more dynamic, human-like, cooperative, and self-directed behavior than a single
monolithic reward.

The thesis sits squarely in the territory that [Softmax](https://softmax.com/) (Emmett Shear's
lab) calls *organic alignment* — studying via multi-agent RL how peers "learn when and how to
share goals, develop specialized roles, and generate collectively intelligent systems." This
survey organizes candidate environments into three tiers, from trivially-evaluable social
dilemmas to open-ended resource worlds, and ends with a **build-vs-reuse** recommendation tuned
to the team's constraint: **Python 3.14, numpy only, no torch/gym installed.**

For each environment: link, what it is, multi-agent?, cooperative/competitive/mixed,
observation/action shapes, install + compute weight, license, and **fit for this project**. A
running flag marks **pure-RL** (reward-driven, no language model) vs **LLM-oriented** (agents are
language models) — the team wants to keep "pure RL agents with reward stacks" cleanly separated
from "LLM agents given rewards."

---

## Tier 1 — Social Dilemmas & Matrix Games

Dead-simple to evaluate: cooperation vs defection is a binary you can score per round. This tier
is where a reward stack first proves itself — e.g. does an innate "reciprocity" or "guilt" drive
shift a population from all-defect toward conditional cooperation? The team's reference point is
the Harvard Business School **"defection game"** (defect = high individual round payoff,
all-cooperate = median payoff, repeated rounds) — a classic iterated prisoner's dilemma (IPD).

### Axelrod-Python — `axelrod` *(pure-RL adjacent / game theory)*
- **Link:** <https://github.com/Axelrod-Python/Axelrod>
- **What:** The definitive IPD research library — 200+ named strategies (Tit-for-Tat and kin),
  head-to-head matches, round-robin tournaments, and Moran-process population dynamics.
- **Multi-agent:** Yes (tournaments of many strategies). **Mixed-motive** (IPD is the canonical
  cooperate-or-defect game).
- **Obs/action:** Not a tensor/space API — actions are binary `C`/`D`; the "observation" is match
  history. Easy to wrap as a gym-style env yourself.
- **Install/weight:** `pip install axelrod`; CPU-only, trivial.
- **License:** MIT.
- **Fit:** **High as a reference and a benchmark of opponents.** Reimplement IPD ourselves (below)
  but borrow Axelrod's strategy zoo as fixed co-players to test whether a reward-stacked agent
  rediscovers Tit-for-Tat-like reciprocity.

### OpenSpiel — `open_spiel` *(pure-RL / game theory)*
- **Link:** <https://github.com/google-deepmind/open_spiel>
- **What:** DeepMind's library of games + algorithms: matrix games, **prisoner's dilemma**, and
  dozens of board/card games, with perfect or imperfect information; n-player, zero-sum,
  cooperative, and general-sum.
- **Multi-agent:** Yes. **All three** (coop, competitive, mixed).
- **Obs/action:** Game-specific information-state / observation tensors + legal-action lists via a
  uniform C++/Python API.
- **Install/weight:** Source build (C++ core); heavier than the pip-only Farama libs. Colab
  notebooks available.
- **License:** Apache-2.0.
- **Fit:** **Medium.** Excellent canonical reference for matrix-game payoff structures and
  game-theoretic baselines (CFR, fictitious play), but the build step is overkill for our
  numpy-only IPD.

### PettingZoo (Classic) — `pettingzoo` *(pure-RL / MARL)*
- **Link:** <https://pettingzoo.farama.org/> · <https://github.com/Farama-Foundation/PettingZoo>
- **What:** The multi-agent counterpart to Gymnasium. Its **Classic** suite has Rock-Paper-Scissors,
  Tic-Tac-Toe, Connect Four, Chess, Go, Hanabi, and poker variants. Two APIs: **AEC** (turn-based)
  and **Parallel** (simultaneous).
- **Multi-agent:** Yes. **Mixed** (varies by game; mostly competitive in Classic).
- **Obs/action:** Per-game; board games use `Discrete` actions with dict observations carrying an
  action mask.
- **Install/weight:** `pip install pettingzoo`; Classic is light.
- **License:** MIT.
- **Fit:** **Medium.** The right *API shape* to copy for our multi-agent interface, and RPS is a
  ready matrix-game sanity check. Adopt the AEC/Parallel mental model even if we don't install it.

> Single-agent baseline worth naming: **Gymnasium** (<https://gymnasium.farama.org/>, MIT, the
> maintained OpenAI Gym successor) — CartPole/MountainCar/Pendulum (classic control) and
> FrozenLake/Taxi/Blackjack (toy text). Pure-RL, `pip install gymnasium`, CPU-trivial. Not
> multi-agent, but the canonical observation/action `Space` API every env above mimics. Useful as
> the smoke-test rig for a single reward-stacked agent before going multi-agent.

---

## Tier 2 — Cooperative Games (Win Together or Not at All)

Here reward is shared or only realized on joint success, so a reward stack is tested on whether
innate drives produce *coordination* and *role specialization* rather than free-riding.

### DeepMind Melting Pot — `dm-meltingpot` *(pure-RL / MARL) — highly relevant*
- **Link:** <https://github.com/google-deepmind/meltingpot>
- **What:** 50+ multi-agent gridworld **substrates** + 256+ held-out test scenarios for evaluating
  generalization to novel social situations and unfamiliar co-players. The canonical
  resource-commons social dilemmas live here: **`commons_harvest`** (sustainable apple harvesting —
  tragedy of the commons; over-harvest kills regrowth) and **`clean_up`** (a public-goods dilemma:
  the river must be cleaned for apples to grow, but cleaning pays the individual nothing).
- **Multi-agent:** Yes (~5–16 agents/substrate). **Mixed-motive** sequential social dilemmas
  spanning cooperation, competition, deception, reciprocation, trust.
- **Obs/action:** Per-agent egocentric **88×88×3 RGB** view (partial observability); **discrete**
  actions (move/turn + substrate-specific, e.g. a "zap" beam).
- **Install/weight:** `pip install dm-meltingpot` (depends on DeepMind Lab2D wheels). Moderate;
  pixels + many agents, GPU helpful. PettingZoo bridge via Farama **Shimmy**.
- **License:** Apache-2.0.
- **Fit:** **Very high as the gold-standard reference.** `commons_harvest` and `clean_up` are
  *exactly* the mixed-motive resource-commons structure our Tier-3 minimal world should emulate.
  Too heavy (pixels, Lab2D) for our numpy-only start, but it defines the target behaviors and gives
  us a published evaluation protocol to grow into.

### Overcooked-AI — `overcooked-ai` *(pure-RL / coordination)*
- **Link:** <https://github.com/HumanCompatibleAI/overcooked_ai>
- **What:** Fully cooperative 2-cook kitchen — fetch ingredients, cook soups, deliver, across
  hand-designed layouts that force tight coordination and on-the-fly task division.
- **Multi-agent:** Yes (2 agents). **Fully cooperative** (shared reward).
- **Obs/action:** **6 discrete actions** (4 moves, interact, stay); observation is a lossless
  `H×W×26` tensor (featurized vector also available).
- **Install/weight:** `pip install overcooked-ai`; very light. (Bundled RL/BC code is deprecated;
  the env itself is maintained.)
- **License:** MIT.
- **Fit:** **High.** The cleanest small cooperative testbed for role specialization and joint-reward
  credit assignment — a good second-week target after IPD, and small enough to reimplement if
  needed.

### Hanabi Learning Environment — `hanabi-learning-environment` *(pure-RL / coop)*
- **Link:** <https://github.com/google-deepmind/hanabi-learning-environment> *(archived, read-only since 2024)*
- **What:** The cooperative imperfect-information card game — a theory-of-mind benchmark where you
  see everyone's hand but your own and communicate only via costly hints.
- **Multi-agent:** Yes (2–5 players). **Fully cooperative.**
- **Obs/action:** ~**658-dim** vectorized observation; single **discrete** index over play/discard/
  hint moves with legal-action masking.
- **Install/weight:** CMake + C++ build; very light at runtime (no pixels).
- **License:** Apache-2.0.
- **Fit:** **Medium.** A great stress test for an innate "communication/curiosity" drive, but the
  partial-information bookkeeping is a lot of surface area early on. Also available via
  PettingZoo-Classic if we want it without the C++ build.

### The Tug/Bungee Cooperative-Control Game — *(to be built; no off-the-shelf env)*
- **What:** The team's flagship physical cooperative task. Three players each hold a rope attached
  to a circular elastic band; they must coordinate individual pulls to triangulate/stretch the
  band, drop it over a bottle, lift, and move the bottle from A to B. Each agent controls only its
  own pull direction/force; **the actuator (band + bottle) is a shared joint effect of all agents'
  actions** — a clean continuous cooperative-*control* problem.
- **Multi-agent:** Yes (3 agents). **Fully cooperative** (the bottle moves only if all three
  coordinate).
- **Obs/action:** Continuous. Per-agent action ≈ `Box(2)` (pull angle + force, or a 2D force
  vector); shared observation ≈ band node positions/tension + bottle pose. A 5–10 node elastic-ring
  spring model with a rigid bottle is a few hundred lines of numpy + simple Euler integration.
- **Install/weight:** N/A — **build it.** numpy-only, CPU, fast-forwardable.
- **License:** Ours.
- **Fit:** **Essential and unique.** No existing env captures "shared actuator, private controls."
  Nearest cousins are PettingZoo-SISL **Multiwalker** (cooperative bipedal carry) and VMAS
  (vectorized continuous MARL) — worth citing as references, but the band-over-bottle mechanic must
  be authored. This is the project's most distinctive continuous-control cooperation signal.

---

## Tier 3 — Open-World Resource Simulators (Top-Down, Speed-Uppable)

Lightweight, procedurally-generated, top-down worlds where agents walk, gather, interact, and
where cooperation or emergent cross-generational goals can arise — the "Minecraft-but-simpler,
top-down, speed-uppable" space the team points at, and exactly what Softmax means by a *resource
simulator*.

### Crafter — `crafter` *(pure-RL / open-ended survival) — very relevant*
- **Link:** <https://github.com/danijar/crafter>
- **What:** 2D open-world survival benchmark — forage, build, craft, fight — scored via 22
  achievements over a 1M-step budget. The reference design for "broad-competence-in-one-cheap-env."
- **Multi-agent:** **Single-agent.**
- **Obs/action:** **64×64×3 RGB** observation; **17 discrete actions**.
- **Install/weight:** `pip install crafter`; very light CPU, deliberately cheap to evaluate.
- **License:** MIT.
- **Fit:** **Very high as the design template** for our minimal top-down world (the tech-tree and
  achievement structure are perfect for a stacked "hunger / safety / curiosity / crafting" reward
  set). Single-agent as shipped — we'd add agents.

### Craftax — `craftax` *(pure-RL / open-ended) — very relevant*
- **Link:** <https://github.com/MichaelTMatthews/Craftax> · paper <https://arxiv.org/abs/2402.16801>
- **What:** Crafter + NetHack-style roguelike rewritten entirely in **JAX** — runs 100–250× faster
  than Crafter (hundreds of thousands of steps/sec on one GPU; 1B-step PPO runs in under an hour).
- **Multi-agent:** **Single-agent** core (gymnax interface); a separate Multi-Agent Craftax exists
  (arXiv 2511.04904).
- **Obs/action:** Craftax-Classic — **1345** symbolic obs / **17** actions; full Craftax — **8268**
  obs / **43** actions (pixel obs also available).
- **Install/weight:** `pip install craftax`; needs JAX (GPU/TPU), slow first JIT then extremely fast.
- **License:** MIT.
- **Fit:** **High for future scaling.** When we need millions of fast rollouts to evolve reward
  stacks across "generations," Craftax is the speed target — but JAX violates the current
  numpy-only constraint, so it's a Phase-2 reuse, not a Phase-1 dependency.

### Neural MMO — `nmmo` *(pure-RL / massively multi-agent) — very relevant*
- **Link:** <https://neuralmmo.github.io> · <https://github.com/NeuralMMO/environment>
- **What:** Massively multi-agent, open-ended MMORPG-like world — procedurally generated maps where
  **128 agents** (1–1024 supported) forage, fight, trade, survive. v2.0 (NeurIPS 2023) adds a
  flexible multi-task system for arbitrary objectives/reward signals.
- **Multi-agent:** Yes (128). **Mixed** (resource competition + optional team/cooperative tasks).
- **Obs/action:** Structured (not pixels) — separate map/inventory/market observations; mixed
  discrete + pointer-network target selection.
- **Install/weight:** `pip install nmmo`; with PufferLib, competent policies trained in ~8h on one
  desktop. Light-to-moderate, CPU-driven, scales with agent count.
- **License:** MIT.
- **Fit:** **High as the multi-agent open-ended reference and Tier-3 scaling target.** Its
  configurable-reward task system is philosophically aligned with a reward stack. Heavier than our
  minimal world, but the closest published analog to "Softmax-style resource simulator at scale."

### MAgent2 — `magent2` *(pure-RL / large-scale gridworld)*
- **Link:** <https://github.com/Farama-Foundation/MAgent2> · <https://magent2.farama.org/>
- **What:** High-performance engine for gridworlds with hundreds-to-thousands of "pixel" agents
  (Battle, Gather, Tiger-Deer, Combined Arms). PettingZoo API.
- **Multi-agent:** Yes (massive). **Competitive/mixed** (Tiger-Deer, Combined Arms add team coop).
- **Obs/action:** Per-agent local feature view; **discrete** move + attack actions.
- **Install/weight:** `pip install magent2`; light per-agent, scales to huge populations.
- **License:** MIT.
- **Fit:** **Medium.** Great when we want *population-scale* emergence (flocking, swarm cooperation)
  rather than rich individual tech-trees. A future stress test for reward-stack scalability.

### Griddly — `griddly` *(pure-RL / configurable gridworld)*
- **Link:** <https://github.com/Bam4d/Griddly> · <https://griddly.readthedocs.io>
- **What:** Highly optimized grid-world engine; games defined in a YAML "GDY" language, GPU-rendered,
  ~70k FPS. Single-agent, multi-agent, and RTS interfaces.
- **Multi-agent:** Yes. **Configurable** (author-defined).
- **Obs/action:** Vectorized one-hot grids / rendered sprites / state maps / event history;
  fully customizable actions.
- **Install/weight:** `pip install griddly` (pre-built binaries); very light.
- **License:** MIT.
- **Fit:** **Medium.** If we ever want a fast, declaratively-specified gridworld without writing the
  engine, Griddly is the best off-the-shelf option — but YAML-GDY means giving up the tight Python
  control over the reward stack we want in Phase 1.

### Lux AI (Season 2 / 3) — `luxai_s2` / `luxai_s3` *(pure-RL / competitive resources)*
- **Link:** <https://github.com/Lux-AI-Challenge/Lux-Design-S2> · <https://lux-ai.org>
- **What:** Competitive 1v1 resource-gathering/optimization on procedurally generated maps; each
  side commands many units and factories.
- **Multi-agent:** Yes (1v1, many units/side). **Competitive.**
- **Obs/action:** Structured game-state obs; multi-discrete per-unit actions. S2 has a JAX version
  (`juxai-s2`).
- **Install/weight:** `pip install luxai_s2` (Python <3.11); moderate, JAX/GPU optional.
- **License:** Apache-2.0.
- **Fit:** **Low-medium.** Resource-rich and procedurally generated, but framed as adversarial 1v1
  and bound to Python <3.11 — off-axis for our cooperation focus and 3.14 target.

### MiniGrid / MiniWorld / BabyAI — `minigrid` *(pure-RL / goal-oriented; BabyAI → LLM)*
- **Link:** <https://minigrid.farama.org/> · <https://github.com/Farama-Foundation/Minigrid>
- **What:** Lightweight goal-oriented grids (2D MiniGrid, 3D MiniWorld); BabyAI adds synthetic
  natural-language instructions ("put the red ball next to the box").
- **Multi-agent:** **No** (single-agent). N/A coop/comp.
- **Obs/action:** Partial **(7,7,3)** symbolic image + mission string; ~7 **discrete** actions.
- **Install/weight:** `pip install minigrid`; extremely light (MiniWorld heavier — 3D).
- **License:** Apache-2.0.
- **Fit:** **Low for multi-agent**, but a useful single-agent rig for debugging a reward stack on a
  navigation task. BabyAI's language missions are a bridge to LLM-agent experiments later.

### LLM-Agent Simulators (kept separate from pure-RL)
- **Stanford Generative Agents** — <https://github.com/joonspk-research/generative_agents>
  (Apache-2.0). "Smallville" sandbox of 25 believable agents. **LLM-oriented** (needs an OpenAI key);
  social/mixed interaction, **no RL reward/observation interface.**
- **AI Town (a16z)** — <https://github.com/a16z-infra/ai-town> (MIT). Deployable town of chatting
  AI characters; defaults to local `llama3` via Ollama. **LLM-oriented**, TypeScript/Convex.
- **Fit:** **Reference only, and explicitly the other branch.** These are the "LLM agents given a
  world" comparison the team wants to *contrast against* pure RL agents with reward stacks — useful
  as a north star for emergent social behavior, not as a Phase-1 RL substrate.

---

## Build vs Reuse Recommendation

**Constraint:** Python 3.14, **numpy only** — no torch, no gym, no JAX, no C++ build step. That
single fact rules out *installing* almost every environment above as a Phase-1 dependency
(Gymnasium/PettingZoo pull in heavy deps; Melting Pot needs Lab2D; Craftax/Lux need JAX; Hanabi/
OpenSpiel need C++ builds). It also aligns with the team's real goal: **full, transparent control
over the reward stack**, which is far easier in a few hundred lines of owned numpy than threaded
through someone else's reward plumbing.

**Therefore: reimplement minimal versions of three core environments, using the libraries above as
behavioral references and as scaling targets for later.**

1. **Iterated Prisoner's Dilemma / "defection game" (Tier 1).** ~100 lines of numpy. Payoff matrix
   with defect = high individual round payoff, all-cooperate = median; N-player and 2-player
   variants; repeated rounds with history. **Reference:** Axelrod-Python's strategy zoo (drop in
   Tit-for-Tat, Grim, Pavlov as fixed co-players) and PettingZoo's AEC/Parallel API shape. This is
   where a reward stack first earns its keep — does an innate reciprocity/guilt drive move the
   population off all-defect?

2. **Top-down resource gridworld, Crafter/commons-harvest style (Tier 3).** ~400–800 lines of
   numpy. Procedural grid, regenerating resources (apples/trees/water), a small craft/tech tree,
   multiple agents with partial local views and discrete move/gather/interact actions. Bake in the
   **tragedy-of-the-commons** dynamic (over-harvest kills regrowth) so cooperation is *measurable*.
   **References:** Crafter (achievement/tech-tree design), Melting Pot `commons_harvest`/`clean_up`
   (the mixed-motive resource-commons structure + a published eval protocol), Neural MMO
   (configurable per-agent reward tasks = a reward stack). Scale-out path later: Craftax (speed),
   Neural MMO (population), MAgent2 (swarm).

3. **Tug/bungee cooperative-control game (Tier 2).** ~200–400 lines of numpy. Elastic-ring spring
   model (5–10 nodes) + rigid bottle + simple Euler integration; 3 agents each emit a 2D pull;
   shared actuator, private controls; reward on lifting and translating the bottle A→B. **No
   off-the-shelf equivalent exists** — nearest references are PettingZoo-SISL Multiwalker and VMAS
   for continuous cooperative control. This is the project's most distinctive signal and must be
   authored.

### Tier → action map

| Tier | Reuse / reference | Build minimal (numpy) |
|---|---|---|
| **1 — Social dilemmas** | Axelrod-Python (strategy zoo), OpenSpiel (matrix payoffs), PettingZoo API shape | **Iterated PD / defection game** — payoff matrix + repeated rounds + history |
| **2 — Cooperative** | Overcooked-AI, Hanabi, Melting Pot (coop substrates); Multiwalker/VMAS for continuous control | **Tug/bungee** continuous cooperative-control game (shared actuator, private pulls) |
| **3 — Open-world resource** | Crafter (design), Melting Pot `commons_harvest`/`clean_up` (eval), Neural MMO (reward tasks); Craftax / MAgent2 (scale-out) | **Top-down resource gridworld** with commons dynamics + small tech tree |

**Sequence:** build IPD first (instant evaluability, proves the reward-stack machinery) → the
resource gridworld (the richest test of emergent cooperation) → the tug/bungee game (the unique
continuous-control contribution). Keep all three behind one small PettingZoo-shaped Parallel API so
agents and the reward stack are environment-agnostic, and keep the door open to *adding* Melting
Pot / Neural MMO / Craftax as installable substrates once a torch/JAX environment is acceptable.

---

### Sources

- Gymnasium — <https://gymnasium.farama.org/> · <https://github.com/Farama-Foundation/Gymnasium>
- PettingZoo — <https://pettingzoo.farama.org/> · <https://github.com/Farama-Foundation/PettingZoo>
- OpenSpiel — <https://github.com/google-deepmind/open_spiel>
- Axelrod-Python — <https://github.com/Axelrod-Python/Axelrod>
- Melting Pot — <https://github.com/google-deepmind/meltingpot>
- Neural MMO — <https://neuralmmo.github.io> · <https://github.com/NeuralMMO/environment> · <https://arxiv.org/abs/2311.03736>
- Hanabi Learning Environment — <https://github.com/google-deepmind/hanabi-learning-environment>
- Overcooked-AI — <https://github.com/HumanCompatibleAI/overcooked_ai>
- Crafter — <https://github.com/danijar/crafter>
- Craftax — <https://github.com/MichaelTMatthews/Craftax> · <https://arxiv.org/abs/2402.16801>
- Griddly — <https://github.com/Bam4d/Griddly> · <https://griddly.readthedocs.io>
- MiniGrid / MiniWorld / BabyAI — <https://minigrid.farama.org/> · <https://github.com/Farama-Foundation/Minigrid>
- MAgent2 — <https://github.com/Farama-Foundation/MAgent2> · <https://magent2.farama.org/>
- Lux AI — <https://github.com/Lux-AI-Challenge/Lux-Design-S2> · <https://lux-ai.org>
- Generative Agents — <https://github.com/joonspk-research/generative_agents>
- AI Town — <https://github.com/a16z-infra/ai-town>
- Byrnes / steering subsystem — <https://www.alignmentforum.org/posts/hE56gYi5d68uux9oM/intro-to-brain-like-agi-safety-3-two-subsystems-learning-and>
- Softmax (organic alignment) — <https://softmax.com/>
