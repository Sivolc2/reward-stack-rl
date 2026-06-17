"""Generational drift over reward stacks.

The team's longer-term idea is that the *weights/parameters of the drive stack*
should drift across generations, the way evolution tunes innate drives. This
module gives a minimal, dependency-free genetic loop over a small "genome" of
drive parameters, with fitness measured by an environment-supplied evaluation
function. It is intentionally generic: a genome is just a dict of floats and a
``build`` callback turns a genome into a configured agent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Genome:
    genes: dict[str, float]

    def mutate(self, rng: np.random.Generator, *, rate: float, scale: float) -> "Genome":
        child = dict(self.genes)
        for k in child:
            if rng.random() < rate:
                child[k] = float(max(0.0, child[k] + rng.normal(0.0, scale)))
        return Genome(child)


def evolve(
    seed_genome: Genome,
    fitness_fn: Callable[[Genome, int], float],
    *,
    generations: int = 8,
    population: int = 12,
    elite_frac: float = 0.34,
    mutation_rate: float = 0.4,
    mutation_scale: float = 0.15,
    seed: int = 0,
) -> tuple[Genome, list[dict]]:
    """Run a simple (mu, lambda)-style evolutionary loop.

    Returns the best genome found and a per-generation history of summary stats
    (mean/best fitness, and the elite-mean of every gene so drift is visible).
    """
    rng = np.random.default_rng(seed)
    pop = [seed_genome.mutate(rng, rate=1.0, scale=mutation_scale) for _ in range(population)]
    history: list[dict] = []
    n_elite = max(1, int(population * elite_frac))

    for gen in range(generations):
        scores = np.array([fitness_fn(g, seed + gen * 1000 + i) for i, g in enumerate(pop)])
        order = np.argsort(scores)[::-1]
        elite = [pop[i] for i in order[:n_elite]]
        elite_scores = scores[order[:n_elite]]

        gene_means = {
            k: float(np.mean([g.genes[k] for g in elite])) for k in seed_genome.genes
        }
        history.append(
            {
                "generation": gen,
                "best_fitness": float(scores.max()),
                "mean_fitness": float(scores.mean()),
                "elite_mean_fitness": float(elite_scores.mean()),
                "gene_means": gene_means,
            }
        )

        # Reproduce: elites survive, offspring are mutated elites.
        children = list(elite)
        while len(children) < population:
            parent = elite[int(rng.integers(len(elite)))]
            children.append(parent.mutate(rng, rate=mutation_rate, scale=mutation_scale))
        pop = children

    best = max(pop, key=lambda g: fitness_fn(g, seed + 99999))
    return best, history
