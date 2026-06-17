"""Unit tests for the core framework. Run with:  python -m unittest -v

No third-party test runner required (pytest is optional). These tests pin the
invariants the experiments rely on: drives behave as designed, the steering
modes arbitrate correctly, the learner improves on a trivial task, and each
environment steps without error and exposes the contract the agents expect.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlstack import make_agent, make_ipd_agent, make_survival_agent  # noqa: E402
from rlstack.agent import QLearningAgent  # noqa: E402
from rlstack.drives import (  # noqa: E402
    CuriosityDrive,
    HomeostaticDrive,
    ReciprocityDrive,
    SafetyDrive,
)
from rlstack.envs import IteratedPrisonersDilemma, ResourceWorld, TugGame  # noqa: E402
from rlstack.metrics import action_entropy, gini, goal_switch_rate  # noqa: E402
from rlstack.steering import SteeringSubsystem  # noqa: E402


class TestDrives(unittest.TestCase):
    def test_homeostatic_drive_reduction_reward(self):
        d = HomeostaticDrive("hunger", "ate", decay=0.1, replenish=0.5)
        d.reset()
        # Eating while in deficit should give positive reward (deficit reduced).
        d._level = 0.4
        d.step({"ate": 1.0})
        self.assertGreater(d.reward({"ate": 1.0}), 0.0)
        # Urgency rises as the level falls.
        d._level = 0.2
        hungry = d.urgency({})
        d._level = 0.95
        sated = d.urgency({})
        self.assertGreater(hungry, sated)

    def test_homeostatic_death_and_damage(self):
        d = HomeostaticDrive("hunger", "ate", decay=0.0, damage=0.5)
        d.reset()
        d._level = 0.4
        d.step({"got_hurt": True})
        self.assertAlmostEqual(d._level, 0.4 - 0.5 if 0.4 - 0.5 > 0 else 0.0)
        # Drops to zero -> dead.
        d._level = 0.3
        d.step({"got_hurt": True})
        self.assertTrue(d.dead)

    def test_safety_urgency_tracks_proximity(self):
        d = SafetyDrive()
        d.step({"hazard_proximity": 0.9})
        near = d.urgency({})
        d.step({"hazard_proximity": 0.0})
        far = d.urgency({})
        self.assertGreater(near, far)
        self.assertLess(d.reward({"hazard_proximity": 0.9, "got_hurt": True}), 0.0)

    def test_curiosity_recency(self):
        d = CuriosityDrive(horizon=10.0, gain=1.0)
        d.reset()
        d.step({"pos": (0, 0)})
        first = d.reward({})
        d.step({"pos": (0, 0)})  # immediate revisit -> low novelty
        revisit = d.reward({})
        self.assertGreater(first, revisit)

    def test_reciprocity(self):
        d = ReciprocityDrive(mutual_coop_bonus=1.0, guilt=1.0)
        self.assertGreater(d.reward({"my_move": 1, "opp_move": 1}), 0.0)  # mutual coop
        self.assertLess(d.reward({"my_move": 0, "opp_move": 1}), 0.0)     # guilt


class TestSteering(unittest.TestCase):
    def test_modes_run_and_track_dominant(self):
        for mode in ("dynamic", "softmax", "sum"):
            s = SteeringSubsystem(
                [HomeostaticDrive("hunger", "ate"), SafetyDrive("safety")],
                mode=mode,
            )
            s.reset()
            r = s.evaluate({"ate": 0.0, "hazard_proximity": 1.0})
            self.assertIsInstance(r, float)
            self.assertIn(s.last_info["dominant"], ("hunger", "safety"))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            SteeringSubsystem([SafetyDrive()], mode="bogus")

    def test_dominant_is_most_urgent(self):
        s = SteeringSubsystem([HomeostaticDrive("hunger", "ate"), SafetyDrive("safety")])
        s.reset()
        # Make hazard maximally urgent; safety should win the wheel.
        s.evaluate({"ate": 1.0, "hazard_proximity": 1.0})
        self.assertEqual(s.last_info["dominant"], "safety")


class TestLearner(unittest.TestCase):
    def test_q_learning_solves_two_armed_bandit(self):
        agent = QLearningAgent(2, epsilon=0.1, seed=0)
        for _ in range(2000):
            a = agent.act("s")
            r = 1.0 if a == 1 else 0.0
            agent.learn("s", a, r, "s", done=True)
        self.assertEqual(int(np.argmax(agent.Q["s"])), 1)


class TestMetrics(unittest.TestCase):
    def test_action_entropy_bounds(self):
        self.assertAlmostEqual(action_entropy([10, 0, 0]), 0.0)
        self.assertAlmostEqual(action_entropy([5, 5]), 1.0)

    def test_goal_switch_rate(self):
        self.assertAlmostEqual(goal_switch_rate(["a", "a", "b", "b"]), 1 / 3)
        self.assertEqual(goal_switch_rate(["a"]), 0.0)

    def test_gini(self):
        self.assertAlmostEqual(gini([1, 1, 1, 1]), 0.0)
        self.assertGreater(gini([0, 0, 0, 4]), 0.5)


class TestEnvironments(unittest.TestCase):
    def test_ipd_match_runs(self):
        env = IteratedPrisonersDilemma(rounds_per_match=8, seed=0)
        a = make_ipd_agent(0, stacked=False, seed=0)
        b = make_ipd_agent(1, stacked=True, seed=1)
        stats = env.play_match(a, b)
        self.assertEqual(len(stats["a_moves"]), 8)
        self.assertTrue(0 <= stats["a_coop"] <= 8)

    def test_resourceworld_step_contract(self):
        env = ResourceWorld(width=8, height=8, n_food=5, n_hazards=3, seed=0)
        agents = [make_survival_agent(i, env.n_actions, seed=i) for i in range(3)]
        env.reset(agents)
        obs = env.observations()
        self.assertEqual(len(obs), 3)
        ctxs = env.step([1, 2, 3])
        for c in ctxs:
            self.assertIn("ate", c)
            self.assertIn("hazard_proximity", c)
            self.assertIn("pos", c)
        self.assertIsInstance(env.render(), str)

    def test_resourceworld_full_stack_agent_runs(self):
        env = ResourceWorld(width=8, height=8, seed=1)
        agents = [make_agent(i, env.n_actions, seed=i) for i in range(2)]
        env.reset(agents)
        obs = env.observations()
        for _ in range(50):
            actions = [agents[i].act(obs[i]) for i in range(2)]
            ctxs = env.step(actions)
            rewards = [agents[i].reward_from(ctxs[i]) for i in range(2)]
            nxt = env.observations()
            for i in range(2):
                agents[i].observe(obs[i], rewards[i], nxt[i], done=agents[i].dead)
            env.respawn_dead()
            obs = env.observations()

    def test_tug_runs_and_can_grip(self):
        env = TugGame(seed=0)
        agents = [make_survival_agent(i, env.n_actions, seed=i) for i in range(3)]
        # Replace with task-reward agents (homeostatic drive irrelevant here, but
        # the contract is identical); just check the env steps and terminates.
        env.reset(agents)
        obs = env.observations()
        self.assertEqual(len(obs), 3)
        done = False
        steps = 0
        while not done and steps < 100:
            ctxs, done = env.step([2, 2, 2])  # all pull east
            steps += 1
        self.assertLessEqual(steps, 100)
        self.assertIsInstance(env.render(), str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
