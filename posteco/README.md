# posteco — a mechanism sandbox for the post-labor transition

The economic application of the reward stack, built to be a **quantitative
mechanism model** for the *"Potholes on the Way to Utopia"* framework: as AI
automates labor, *which institutions* turn the pothole into a sinkhole, and which
keep humans aboard the flywheel?

It is **not a forecast.** It sweeps an automation level and answers a conditional
question ("*given* automation A, which regime wins?"). It cannot say whether or
when automation rises — Robin Hanson's timing objection is respected (see Caveats).

```bash
python -m posteco.experiments.run_all        # all 3 experiments, ~2s, numpy only
```

## The thesis, as mechanism

After the essay: **supply solves itself** (automation makes goods cheap), but
**demand does not** — realized output is `min(capacity, effective demand)`. If
income concentrates, the rich hoard, demand collapses, and capacity sits idle
("too much of everything and can't move the inventory"). As automation `A → 1`,
the **labor share of output → 0 and the capital share → 1** (NAPCS → 1): "who eats
collapses to who owns the machine," and redistribution that runs through *wages*
stops working.

The reward-stack twist: each agent's propensity to consume / hoard / invest is set
by its drives (subsistence vs acquisition) **modulated by fear vs vision** (`p_good`).
Fear amplifies hoarding → demand collapse (Moloch); a credible shared vision flips
cooperative spending into the individually rational move. This is the steering
subsystem at civilizational scale.

## The three pillars as levers (`Regime` in economy.py)

| Pillar | Lever | Mechanism |
|---|---|---|
| 1 · **Charter** | `charter_epsilon` | a non-dilutable share of *capital income* paid out per-capita. At the formation/capital layer, so **not gameable**. |
| — · status quo | `tax_rate`, `capital_evasion`, `tax_leak` | Piketty-style income tax — **leaky**: capital routes around it, so it weakens exactly as capital's share grows. |
| 2 · **Advocates** | `advocate_coverage`, `advocate_mpc_boost` | demand-side power: surfaces latent wants so income gets *spent* not hoarded ("demand must find new places to be put"). |
| 3 · **Vision** | `vision_floor` / `fix_belief` | the Schelling-point signal `p_good` that flips agents from defensive hoarding to cooperative investment. |

## What the three experiments show (representative seeded run)

**1 · The pothole is optional** (`exp_automation_sweep`). The automation level at
which the economy tips into the sinkhole (mass subsistence failure + demand
collapse) depends on the regime:

```
   laissez-faire   tips at A ≈ 0.85
   wealth-tax      tips at A ≈ 0.90   (leaky — capital escapes; only delays it)
   charter         never tips across the sweep
At A=0.90:  laissez-faire 94% fail / 51% capacity used;  charter 0% fail / 100% used
```

**2 · How big must the commons pool be?** (`exp_charter_threshold`). The charter
prevents the sinkhole once **ε ≈ 0.10**; the framework's ~20% has margin. **But**
the wealth Gini barely moves even at ε=0.50 (0.97 → 0.91): the charter **stops
starvation without de-concentrating capital** — it redistributes the *flow*, not
the *stock*. It changes who eats, not who owns the machine. (This is
Schmachtenberger's Game B challenge, quantified and respected.)

**3 · The fear that creates what it fears** (`exp_fear_vs_vision`). Holding the
economy and institutions fixed, varying only shared belief: **pure fear idles ~24%
of capacity; vision cuts it to ~2%.** Same factories, same rules — ~22 points of
prosperity created or destroyed purely by what people believe. "You cannot make
things without customers"; fear makes people stop being customers. This is why
vision is load-bearing.

## Caveats (read these — the model is honest about its limits)

- **Conditional, not predictive.** It sweeps `automation 0→1`, a variable Hanson
  argues is nowhere near moving (automation is <5% of world income; his ~80%
  "game over" threshold he puts *centuries* out). The model answers "*if* it rises,
  which regime wins?" — never *whether* or *when*.
- **Mechanism, not magnitude.** ε, evasion, `p_good` are illustrative, not
  estimated. Read the *directions and tipping structure*, never a Gini as measured.
- **The Game B limit** is built in and reported, not hidden: fixed-reward-stack
  agents in a rivalrous system; the model quantifies distribution, not whether the
  regime is genuinely anti-rivalrous.
- **Belief follows outcomes here**; the essay's stronger claim — that vision enables
  the *political adoption* of the charter — is a political-economy mechanism this
  ABM does not model (Experiment 3 isolates only the demand channel, via a pinned
  belief).
- **Reward-stack reductionism.** Representing people as a stack of reward functions
  cannot capture preference *discovery* — the very thing Pillar 2 (advocates) is
  about. `advocate_mpc_boost` is a crude stand-in.

See `../RESEARCH.md` for the reward-stack lineage and `economy.py` for the model.
