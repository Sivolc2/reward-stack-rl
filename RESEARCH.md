---
title: "Reward-Stack RL — Literature & Landscape Review"
date: 2026-06-17
tags: [reinforcement-learning, multi-agent-rl, intrinsic-motivation, homeostatic-rl, brain-like-agi, reward-design, open-endedness, cooperative-ai]
---

# Reward-Stack RL — Literature & Landscape Review

## Framing the hypothesis

Most RL systems optimize a single scalar reward. Brains, by contrast, appear to be steered by *many* genetically-specified drives — hunger, fear, curiosity, social affiliation, pain avoidance — each tracked separately, each waxing and waning in urgency, and arbitrated moment-to-moment into a single stream of behavior. This project asks whether a **stack of many interacting reward functions, with a steering/arbitration layer and dynamically-changing weights, can reproduce the self-directing, "human-like" behavior of brains** better than one monolithic reward — and whether such an architecture, tested in multi-agent grid and social environments, naturally yields agents that set their own goals, cooperate, and remain curious without harming others. The literature below spans the project's intellectual lineage (neuroscience-grounded AGI theory) and the concrete technical building blocks (intrinsic motivation, homeostatic RL, multi-objective and modular RL, cooperative MARL, open-ended/evolutionary RL). Each section ends with a **"How it maps to this project"** note.

---

## 1. Intellectual lineage: brains as a stack of reward functions

### Steven Byrnes — Brain-Like-AGI Safety

Byrnes' [*Intro to Brain-Like-AGI Safety*](https://www.lesswrong.com/s/HzcM2dkCq7fwXBej8) series is the project's central anchor. His core decomposition: the brain is a **Steering Subsystem** (roughly the hypothalamus and brainstem, "<10%" of the brain, running genetically-hardwired "business logic") plus a **Learning Subsystem** (cortex, hippocampus, striatum, amygdala, cerebellum, ">90%", a from-scratch learning algorithm that begins with random parameters). See [Post 3, "Two subsystems: Learning & Steering"](https://www.lesswrong.com/posts/hE56gYi5d68uux9oM/intro-to-brain-like-agi-safety-3-two-subsystems-learning-and). Critically, **the Learning Subsystem "has no goals or values"** of its own; all motivation originates in the Steering Subsystem's **innate drives** (which Byrnes also calls "primary rewards") — "the root cause of why some things are inherently motivating / appetitive and other things are inherently demotivating / aversive."

In [Post 6, "Big picture of motivation, decision-making, and RL"](https://www.lesswrong.com/posts/qNZSBqLEh4qLRqgWW/intro-to-brain-like-agi-safety-6-big-picture-of-motivation), Byrnes introduces **Thought Assessors** — supervised-learning predictors that take a "thought" from the thought generator and guess what Steering-Subsystem signals it will lead to. There are "perhaps hundreds to thousands of them," and a distinguished one computes **valence** ("how rewarding it is to the animal"). [Post 7, "From hardcoded drives to foresighted plans"](https://www.alignmentforum.org/posts/zXibERtEWpKuG5XAC/intro-to-brain-like-agi-safety-7-from-hardcoded-drives-to) gives the worked example the project cites: a hardwired brainstem circuit (e.g. detecting visual "skittering") triggers a primary reward/punishment, which the Learning Subsystem then **symbol-grounds** into foresighted, abstract motivations (fear-of-creature). This is exactly the "stack of innate drives, wired by evolution, nested under survival" picture the project wants to build computationally.

### Adam Marblestone — diverse, internally-generated cost functions

Marblestone, Wayne & Kording's [*Toward an Integration of Deep Learning and Neuroscience*](https://pmc.ncbi.nlm.nih.gov/articles/PMC5021692/) (Frontiers in Computational Neuroscience, 2016; [arXiv:1606.03813](https://arxiv.org/pdf/1606.03813)) advances three hypotheses: **(1)** the brain optimizes cost functions; **(2)** these cost functions are **diverse and heterogeneous, differing across brain areas and changing over development**; and **(3)** optimization happens within a pre-structured architecture matched to the computational problems behavior poses. Hypothesis 2 is the direct neuroscientific charter for a *stack* of separately-defined rewards rather than one global loss, and the emphasis that cost functions are largely **internally generated** (not handed to the brain by the environment) connects forward to intrinsic-motivation and "where do rewards come from" work below.

> **How it maps to this project:** Implement the Steering/Learning split literally — a small, hand-coded module emitting many drive signals, feeding a from-scratch learner that has no intrinsic goals. Treat each drive as one of Marblestone's heterogeneous, internally-generated cost functions. The "valence" arbitrator becomes the project's steering layer that collapses the drive stack into the scalar the policy actually maximizes.

---

## 2. Intrinsic motivation / curiosity-driven RL

This is the computational analog of a **curiosity drive** in the stack. Schmidhuber's [*Formal Theory of Creativity, Fun, and Intrinsic Motivation (1990–2010)*](https://people.idsia.ch/~juergen/ieeecreative.pdf) gives intrinsic reward as *compression/prediction progress*: reward the learner for improvements in its own predictor/compressor of the action-observation history — the formal version of "novelty is fun." Pathak et al.'s [Intrinsic Curiosity Module (ICM, ICML 2017)](https://arxiv.org/abs/1705.05363) operationalizes curiosity as the agent's *prediction error* in a self-supervised inverse-dynamics feature space; an agent rewarded only by curiosity learns to explore VizDoom and Super Mario with no extrinsic reward. Burda et al.'s [Random Network Distillation (RND, 2018)](https://arxiv.org/abs/1810.12894) gives a simpler, more robust novelty bonus — prediction error against a fixed random network — and was the first method to beat average human performance on Montezuma's Revenge without demonstrations.

Two complementary strands matter. **Empowerment** (Klyubin, Polani & Nehaniv; Salge et al., [*Empowerment — An Introduction*](https://link.springer.com/chapter/10.1007/978-3-642-53734-9_4)) is an information-theoretic intrinsic drive: maximize the channel capacity between an agent's actions and its future sensor states — i.e. *seek states where you have the most control*. This is a strikingly good formal candidate for a "self-preservation / agency" drive. Count-based exploration and the broader family are surveyed in [Aubret et al., *A survey on intrinsic motivation in RL*](https://arxiv.org/pdf/1908.06976).

> **How it maps to this project:** Add at least one ICM- or RND-style **curiosity drive** to the stack (RND is the cheaper, more stable default). An **empowerment** term is a principled "stay-in-control / self-preservation" drive that ties directly to the project's end-goal of an agent that wants to *predict universe states* (knowledge-seeking) while keeping its own options open. These should be *separate tracked signals*, not pre-summed into the extrinsic reward.

---

## 3. Homeostatic / drive-based RL

This is the computational analog of **nested survival drives with dynamic weights**. Keramati & Gutkin's [*Homeostatic reinforcement learning for integrating reward collection and physiological stability*](https://elifesciences.org/articles/04811) (eLife, 2014) is the key paper: the agent carries internal physiological state variables (energy, hydration), each with a **homeostatic setpoint**, and the **rewarding value of an outcome is defined as the reduction in the distance between internal state and setpoint** ("drive reduction"). They prove that reward-seeking is then *equivalent* to defending physiological stability — a clean, biologically-grounded formalism descending from Hull's mid-century drive-reduction theory. Crucially, a drive's *urgency rises as its variable departs setpoint*, so the same outcome (eating) is highly rewarding when hungry and worthless when sated — i.e. **dynamic weights fall out of the formalism for free**. The line continues in recent work, e.g. [Yoshida & colleagues / *Emergence of integrated behaviors through direct optimization for homeostasis*](https://www.sciencedirect.com/science/article/pii/S0893608024003034) and [*Linking Homeostasis to RL: Internal State Control of Motivated Behavior*](https://arxiv.org/pdf/2507.04998).

> **How it maps to this project:** Implement each survival drive as a **setpoint-error signal**: `drive_urgency = f(|internal_var − setpoint|)`. Reward for a drive = reduction in that error. This gives the project its **dynamic weights** organically — no hand-tuned schedule needed — and gives a principled baseline (a single "survival/energy" drive) against which the multi-drive stack is compared.

---

## 4. Multi-objective, multi-reward, and modular RL

This literature clarifies the project's *central distinction*: "one reward with many summed terms" vs. "a dynamic stack of separately-tracked rewards with a steering/arbitration layer." **Multi-objective RL (MORL)** treats reward as a *vector*; there is no total order over policies, only a [Pareto front](https://jmlr.org/papers/volume15/vanmoffaert14a/vanmoffaert14a.pdf). The standard collapse is **scalarization**, but **linear scalarization provably cannot recover non-convex regions of the Pareto front** — a known limitation motivating non-linear (Chebyshev/Tchebysheff) scalarization. This is a direct technical warning: naively summing the drive stack into one number is *lossy* and biases toward "convex compromise" behaviors. **Potential-based reward shaping** (Ng et al.) is the safe way to add shaping terms *without* changing the optimal policy — useful if drives are added as guidance rather than as genuine objectives.

The more apt framing is **modular RL**: decompose a multi-goal problem into simultaneously-running single-goal subagents that **share an action set but each have their own reward signal and value function**, then have an **arbitrator** combine their per-action preferences (Q-values) — either *hard selection* (one module wins) or *soft fusion* (weighted vote). Doya et al.'s **Multiple Model-based RL (MMRL)** ([Multiple Model-Based Reinforcement Learning](https://www.researchgate.net/publication/11352258_Multiple_Model-Based_Reinforcement_Learning)) is the canonical instance: each module has a predictive model, and a **"responsibility signal" — a softmax over modules' prediction errors — gates both learning and action selection**. This is essentially a learnable steering layer.

> **How it maps to this project:** This is the project's architectural core. Build drives as **modular value functions, each with its own reward**, and a **steering/arbitration layer** that combines them. Doya's softmax-responsibility is the natural template: arbitrate by **softmax over drive-urgency** (or over per-drive Q-values weighted by urgency). Heed the MORL warning — keep drives vector-valued internally and treat scalarization as a *deliberate, possibly non-linear, possibly learned* arbitration step, not an accidental sum.

---

## 5. Cooperative multi-agent RL & social dilemmas

Because the project tests in *multi-agent* social environments, the cooperation literature is central. Leibo et al.'s [*Multi-agent RL in Sequential Social Dilemmas*](https://arxiv.org/pdf/1702.03037) (2017) generalizes the iterated Prisoner's Dilemma — and Axelrod's classic tournaments where **tit-for-tat** prevailed — into **temporally-extended grid environments** where cooperativeness is a property of *policies*, not single moves. DeepMind's [**Melting Pot**](https://deepmind.google/blog/melting-pot-an-evaluation-suite-for-multi-agent-reinforcement-learning/) (Leibo et al., 2021; [Melting Pot Contest, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/1d3ea22480873b389a3365d711eb1e91-Paper-Datasets_and_Benchmarks_Track.pdf)) is the standard benchmark: ~21 substrates and 85+ test scenarios evaluating *social* generalization — how agents cooperate with novel, unseen partners. Cooperative MARL training methods (value-decomposition like **QMIX**, centralized-critic actor-critic like **MADDPG**) provide the optimization machinery.

Most relevant as *parallel work*: **Softmax** (the SF lab founded March 2025 by Emmett Shear, Adam Goldstein, and David Bloomin; [team page](https://softmax.com/team), [LessWrong overview](https://www.lesswrong.com/posts/QGQiCuE33iHFv9jkv/softmax-emmett-shear-s-new-ai-startup-focused-on-organic)). Softmax is building a mathematical theory of **"organic alignment"** — AI systems that align *with each other and with humans through emergent cooperation rather than hard-coded hierarchical control*, studied via **multi-agent RL**. Shear's claim: "there are general principles that govern the alignment of any group of intelligent learning agents — whether an ant colony, humans on a team, or cells in a body," and AIs should learn to cooperate "just as our cells cooperate to form our bodies." This is the project's nearest neighbor: both bet that alignment/sociality is an *emergent property of many interacting agents with the right internal incentives*, not a single specified objective.

> **How it maps to this project:** Use a **Melting-Pot-style sequential social dilemma** as the testbed substrate; report cooperation with *held-out* partners, not just self-play. A social/affiliation drive in the stack is the lever for studying whether prosocial behavior emerges. Position the work explicitly against Softmax's organic-alignment program (and contrast: the project specifies drives innately, à la Byrnes, whereas Softmax leans on emergent cooperation).

---

## 6. AlphaZero/MuZero vs. open-ended, goal-setting agents

The transcript's "New Zero / Uzero" refers to **AlphaZero** and **MuZero** — and the key observation is correct: these are **adversarial systems with a single win/loss reward by design**. They are the *antithesis* of the project's thesis: one terminal reward, zero-sum, no internal drive structure, no goal-setting. They show how far a *monolithic* reward can go in closed games, which makes them the natural foil.

The contrasting cluster is **open-ended, agent-driven** environments. **Neural MMO** (Suarez et al., [arXiv:1903.00784](https://arxiv.org/abs/1903.00784), [platform paper 2021](https://arxiv.org/abs/2110.07594)) is a persistent, massively-multiagent world with large populations, long horizons, and open-ended tasks — ideal for studying drives and emergent niches at scale. **Generative Agents** (Park et al., [*Interactive Simulacra of Human Behavior*, UIST 2023, arXiv:2304.03442](https://arxiv.org/abs/2304.03442)) populated the "Smallville" sandbox with 25 LLM agents that plan days, form relationships, and self-reflect — a demonstration of believable *self-directed* behavior (though driven by LLM prompting rather than a learned reward stack). **POET** (Wang et al., [*Paired Open-Ended Trailblazer*, GECCO 2019](https://dl.acm.org/doi/pdf/10.1145/3321707.3321799)) and **quality-diversity** algorithms keep an expanding meta-population of agent–environment pairs, co-evolving challenges with solutions — the machinery for "agents that set their own goals in open worlds across generations."

> **How it maps to this project:** Use AlphaZero/MuZero as the **single-reward baseline foil**. Borrow Neural MMO (or a lightweight grid analog) as the **open-ended substrate** where drives can produce emergent niches. POET/QD point toward *generational* experiments where the *environment* and the *drive stack* co-evolve.

---

## 7. Evolutionary / generational reward drift

The project wants reward functions that **drift over generations** — exactly the program of **optimal-reward / reward-search** work. Singh, Lewis & Barto's [*Where Do Rewards Come From?*](https://all.cs.umass.edu/pubs/2009/singh_l_b_09.pdf) (2009) frames an agent's reward function as itself the product of evolution: define an **optimal reward function given a fitness function and a distribution of environments**, then *search* for it. Their striking result — the optimal reward "need not bear a direct relationship to the fitness function" yet outperforms fitness-based reward — is the theoretical justification for *intrinsic* drives (curiosity, drives with no immediate survival payoff) existing at all. Niekum et al.'s [*Evolved Intrinsic Reward Functions for RL*](https://cdn.aaai.org/ojs/7772/7772-13-11301-1-2-20201228.pdf) uses **genetic programming to search the space of reward-function programs** over the full state space; **Evolutionary RL (ERL)** hybrids and [*Evolution of Rewards for Food and Motor Action by Simulating Birth and Death*](https://arxiv.org/html/2406.15016v1) extend this to populations evolving their reward signals.

This connects to the project's stated end-goal via two further sources. **Era of Experience** (Silver & Sutton, [DeepMind, 2025](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)) argues for agents on **continuous streams of experience** optimizing **grounded rewards** that are *measured by the agent from the environment* rather than hand-specified — i.e. *dynamic, internally-computed* reward, precisely what a homeostatic drive stack produces. And **knowledge-seeking agents** — Orseau's [*Universal Knowledge-Seeking Agents*](https://www.researchgate.net/publication/281723366_Universal_knowledge-seeking_agents) and the AIXI line, recently revisited in DeepMind's [*From AGI to ASI*](https://arxiv.org/abs/2606.12683) (arXiv:2606.12683, 2026) — propose **exploring/predicting all universe states** as a *safer* objective than reward maximization. Together with James Carse's [*Finite and Infinite Games*](https://en.wikipedia.org/wiki/Finite_and_Infinite_Games) (finite games are played to win; **infinite games are played to continue the play**, with rules the players *adjust*), these motivate the project's north star: an agent oriented toward curiosity / universe-state prediction that *plays infinite games* and sets its own goals — with **morality and self-preservation encoded as additional drives in the stack** so the curiosity drive never builds bad states for humans.

> **How it maps to this project:** Treat the **drive weights/setpoints themselves as an evolvable genome**: run generations, select on a fitness signal (survival + social outcomes), and let the *reward stack drift* — the Singh/Niekum program. Use a grounded, Era-of-Experience-style reward (computed from environment, not handed in). Encode the knowledge-seeking objective as a curiosity/empowerment drive **bounded by** explicit morality and self-preservation drives, so "predict the universe" is constrained — the infinite-game agent that doesn't break its world.

---

## Gaps / what's novel here

- **The combination is the novelty.** Each ingredient exists — homeostatic RL, ICM/RND curiosity, modular/MMRL arbitration, social-dilemma MARL, evolved rewards — but **no existing system combines a Byrnes-style innate drive *stack* + a softmax steering/arbitration layer + homeostatic *dynamic* weights + generational drift, evaluated in cooperative multi-agent social dilemmas.**
- **Dynamic, urgency-weighted arbitration over a *labeled* drive stack** is under-explored. MORL scalarizes statically; modular RL arbitrates but rarely with homeostatic urgency as the gating signal; homeostatic RL usually has few drives and no social setting.
- **Innate-and-then-grounded vs. purely emergent.** Unlike Softmax's emergent organic alignment, this project *specifies* drives innately (Byrnes) and lets the Learning Subsystem symbol-ground them — a testable middle path between hand-coded objectives and pure emergence.
- **Knowledge-seeking bounded by moral/self-preservation drives** as *peer terms in the same stack* (not as a constraint bolted on afterward) is a genuinely fresh safety framing.

## Concrete recommendations for our prototype

1. **Architecture:** literal Steering/Learning split — a small hand-coded module emitting many drive signals into a from-scratch learner that has no goals of its own (Byrnes).
2. **Drives as homeostatic setpoint-error signals:** `reward_i = reduction in |internal_var_i − setpoint_i|`; urgency rises with departure from setpoint, giving **dynamic weights for free** (Keramati & Gutkin).
3. **Arbitrate with softmax over drive-urgency** (Doya MMRL "responsibility"), combining per-drive value functions rather than pre-summing rewards into one scalar.
4. **Keep reward vector-valued internally;** if you must scalarize, prefer **non-linear (Chebyshev) scalarization** — linear summing cannot reach non-convex Pareto regions.
5. **Include an explicit curiosity drive** (RND first for stability; ICM as alternative) and an **empowerment drive** as a principled self-preservation/agency term.
6. **Baselines:** (a) single survival/energy reward; (b) one monolithic summed reward of all terms; (c) AlphaZero-style single win/loss in a competitive variant — to isolate what the *stack + steering* buys.
7. **Testbed:** a **Melting-Pot-style sequential social dilemma** on a grid; evaluate cooperation with **held-out partners**, plus an open-ended (Neural-MMO-lite) substrate for emergent niches.
8. **Add a social/affiliation drive** to probe emergent cooperation; relate findings explicitly to Softmax's "organic alignment."
9. **Generational drift:** make drive setpoints/weights an evolvable genome; select on survival + social fitness over generations (Singh/Niekum optimal-reward search).
10. **Encode morality + self-preservation as peer drives** bounding the curiosity/knowledge-seeking term, so the "predict universe states" objective never produces bad states for humans (Orseau / *From AGI to ASI*; Carse infinite games).

---

## References

- Aubret, A., Matignon, L., & Hassas, S. (2019). *A survey on intrinsic motivation in reinforcement learning.* https://arxiv.org/pdf/1908.06976
- Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018). *Exploration by Random Network Distillation.* https://arxiv.org/abs/1810.12894
- Byrnes, S. (2022). *Intro to Brain-Like-AGI Safety* (series). https://www.lesswrong.com/s/HzcM2dkCq7fwXBej8 — esp. [Post 3 (Two subsystems)](https://www.lesswrong.com/posts/hE56gYi5d68uux9oM/intro-to-brain-like-agi-safety-3-two-subsystems-learning-and), [Post 6 (Motivation & RL)](https://www.lesswrong.com/posts/qNZSBqLEh4qLRqgWW/intro-to-brain-like-agi-safety-6-big-picture-of-motivation), [Post 7 (Hardcoded drives → plans)](https://www.alignmentforum.org/posts/zXibERtEWpKuG5XAC/intro-to-brain-like-agi-safety-7-from-hardcoded-drives-to)
- Carse, J. P. (1986). *Finite and Infinite Games.* https://en.wikipedia.org/wiki/Finite_and_Infinite_Games
- Doya, K., Samejima, K., Katagiri, K., & Kawato, M. (2002). *Multiple Model-Based Reinforcement Learning.* https://www.researchgate.net/publication/11352258_Multiple_Model-Based_Reinforcement_Learning
- Genewein, T., Franklin, M., Orseau, L., et al. / Google DeepMind (2026). *From AGI to ASI.* arXiv:2606.12683. https://arxiv.org/abs/2606.12683
- Keramati, M., & Gutkin, B. (2014). *Homeostatic reinforcement learning for integrating reward collection and physiological stability.* eLife. https://elifesciences.org/articles/04811
- Klyubin, A., Polani, D., Nehaniv, C.; Salge, C., et al. *Empowerment — An Introduction.* https://link.springer.com/chapter/10.1007/978-3-642-53734-9_4
- Leibo, J. Z., et al. (2017). *Multi-agent Reinforcement Learning in Sequential Social Dilemmas.* https://arxiv.org/pdf/1702.03037
- Leibo, J. Z., et al. (2021). *Melting Pot* (DeepMind). https://deepmind.google/blog/melting-pot-an-evaluation-suite-for-multi-agent-reinforcement-learning/ ; [Contest, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/1d3ea22480873b389a3365d711eb1e91-Paper-Datasets_and_Benchmarks_Track.pdf)
- Marblestone, A. H., Wayne, G., & Kording, K. P. (2016). *Toward an Integration of Deep Learning and Neuroscience.* Frontiers in Computational Neuroscience. https://pmc.ncbi.nlm.nih.gov/articles/PMC5021692/ ; arXiv:1606.03813 https://arxiv.org/pdf/1606.03813
- Niekum, S., Barto, A. G., & Spector, L. *Evolved Intrinsic Reward Functions for Reinforcement Learning.* https://cdn.aaai.org/ojs/7772/7772-13-11301-1-2-20201228.pdf
- Orseau, L. *Universal Knowledge-Seeking Agents.* https://www.researchgate.net/publication/281723366_Universal_knowledge-seeking_agents
- Park, J. S., et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior.* UIST. arXiv:2304.03442. https://arxiv.org/abs/2304.03442
- Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). *Curiosity-driven Exploration by Self-supervised Prediction.* ICML. https://arxiv.org/abs/1705.05363
- Schmidhuber, J. (2010). *Formal Theory of Creativity, Fun, and Intrinsic Motivation (1990–2010).* https://people.idsia.ch/~juergen/ieeecreative.pdf
- Silver, D., & Sutton, R. S. (2025). *Welcome to the Era of Experience* (DeepMind). https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf
- Singh, S., Lewis, R. L., & Barto, A. G. (2009). *Where Do Rewards Come From?* https://all.cs.umass.edu/pubs/2009/singh_l_b_09.pdf
- Softmax (Shear, Goldstein, Bloomin, 2025). *Organic alignment.* https://softmax.com/team ; https://www.lesswrong.com/posts/QGQiCuE33iHFv9jkv/softmax-emmett-shear-s-new-ai-startup-focused-on-organic
- Suarez, J., et al. (2019/2021). *Neural MMO.* arXiv:1903.00784 https://arxiv.org/abs/1903.00784 ; arXiv:2110.07594 https://arxiv.org/abs/2110.07594
- Van Moffaert, K., & Nowé, A. (2014). *Multi-Objective Reinforcement Learning using Sets of Pareto-Dominating Policies.* JMLR. https://jmlr.org/papers/volume15/vanmoffaert14a/vanmoffaert14a.pdf
- Wang, R., Lehman, J., Clune, J., & Stanley, K. O. (2019). *POET: Paired Open-Ended Trailblazer.* GECCO. https://dl.acm.org/doi/pdf/10.1145/3321707.3321799
