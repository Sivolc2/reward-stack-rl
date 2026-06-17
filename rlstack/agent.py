"""Agents: a tabular Q-learner plus the BrainAgent that wires it to drives.

We deliberately use plain tabular Q-learning (numpy + a dict) rather than deep
RL: the scientific question here is about *reward structure*, not function
approximation, and a tabular learner makes the experiments fast, deterministic
under a seed, and runnable anywhere (Python + numpy, no GPU). The state spaces
of our environments are kept small and discrete on purpose.

``BrainAgent`` is the key abstraction: a Learning Subsystem (the Q-learner) plus
a Steering Subsystem (the drive stack). Crucially the *reward the learner sees
is computed by the agent's own drives* in response to environment events — the
environment emits facts ("you ate", "a hazard is adjacent"), and the innate
steering subsystem decides what those facts are worth. This matches Byrnes'
picture and lets us swap the entire reward structure (single reward vs. dynamic
stack) without touching the environment.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Hashable, Mapping

import numpy as np

from .steering import SteeringSubsystem


class QLearningAgent:
    """Epsilon-greedy tabular Q-learning."""

    def __init__(
        self,
        n_actions: int,
        *,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.9995,
        seed: int = 0,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.Q: dict[Hashable, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )

    def act(self, state: Hashable, *, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.Q[state]
        # Random tie-break so we don't bias toward action 0.
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def learn(
        self,
        state: Hashable,
        action: int,
        reward: float,
        next_state: Hashable,
        done: bool,
    ) -> None:
        q = self.Q[state]
        target = reward if done else reward + self.gamma * self.Q[next_state].max()
        q[action] += self.alpha * (target - q[action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class BrainAgent:
    """A learner + a steering subsystem (stack of drives).

    Parameters
    ----------
    agent_id:
        Identifier used by multi-agent environments.
    n_actions:
        Size of the discrete action space.
    steering:
        The :class:`SteeringSubsystem` holding this agent's drive stack.
    learner_kwargs:
        Passed through to :class:`QLearningAgent`.
    """

    def __init__(
        self,
        agent_id: int,
        n_actions: int,
        steering: SteeringSubsystem,
        **learner_kwargs: Any,
    ) -> None:
        self.id = agent_id
        self.steering = steering
        self.learner = QLearningAgent(n_actions, **learner_kwargs)
        self._last_state: Hashable | None = None
        self._last_action: int | None = None
        # Lightweight behavioural trace for metrics.
        self.action_counts = np.zeros(n_actions, dtype=np.int64)
        self.dominant_history: list[str] = []

    def reset_episode(self) -> None:
        self.steering.reset()
        self._last_state = None
        self._last_action = None

    def act(self, state: Hashable, *, greedy: bool = False) -> int:
        action = self.learner.act(state, greedy=greedy)
        self._last_state = state
        self._last_action = action
        self.action_counts[action] += 1
        return action

    def reward_from(self, ctx: Mapping[str, Any]) -> float:
        """Convert environment events into the agent's own scalar reward."""
        r = self.steering.evaluate(ctx)
        dom = self.steering.last_info.get("dominant")
        if dom is not None:
            self.dominant_history.append(dom)
        return r

    def observe(self, state: Hashable, reward: float, next_state: Hashable, done: bool) -> None:
        if self._last_state is not None and self._last_action is not None:
            self.learner.learn(self._last_state, self._last_action, reward, next_state, done)
        if done:
            self.learner.decay_epsilon()

    @property
    def dead(self) -> bool:
        return self.steering.any_dead()
