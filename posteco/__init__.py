"""posteco — a stylized agent-based model of the post-labor transition.

The economic application of the reward stack. See economy.py and README.md.
A *mechanism* model (conditional on automation), not a forecast.
"""
from .economy import REGIMES, PostLaborEconomy, Regime

__all__ = ["PostLaborEconomy", "Regime", "REGIMES"]
