"""Environments, organised in the three tiers from the design discussion.

Tier 1  social dilemmas / matrix games   -> prisoners.IteratedPrisonersDilemma
Tier 2  cooperative joint-action control  -> tug.TugGame
Tier 3  open-ended resource gathering      -> gridworld.ResourceWorld
"""
from .gridworld import ResourceWorld
from .prisoners import IteratedPrisonersDilemma
from .tug import TugGame

__all__ = ["IteratedPrisonersDilemma", "ResourceWorld", "TugGame"]
