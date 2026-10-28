from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from Agentic_HEMS.shiftable_scheduler import (
    calculate_net_energy_cost,
    schedule_time_shiftable_loads,
)


@dataclass
class HEMSDecision:
    """Result of one household agent's scheduling decision."""

    agent_name: str
    original_cost: float
    optimized_cost: float
    cost_saving: float
    scheduled_data: pd.DataFrame


class RuleBasedHEMSAgent:
    """
    Household agent that schedules time-shiftable appliances.

    This class provides the interface that can later be replaced
    by a reinforcement-learning or LLM-based agent.
    """

    def __init__(
        self,
        agent_name: str,
        metadata: dict[str, Any],
        interval_hours: float = 0.25,
        export_tariff: float = 0.0097,
    ) -> None:
        self.agent_name = agent_name
        self.metadata = metadata
        self.interval_hours = interval_hours
        self.export_tariff = export_tariff

    def make_decision(
        self,
        day_data: pd.DataFrame,
    ) -> HEMSDecision:
        """Create a cost-aware daily appliance schedule."""
        load_columns = self.metadata.get(
            "time_shiftable_loads",
            [],
        )

        prices = pd.to_numeric(
            day_data["ToU"],
            errors="coerce",
        )
        original_grid = pd.to_numeric(
            day_data["grid"],
            errors="coerce",
        )

        original_cost = calculate_net_energy_cost(
            grid_power=original_grid,
            prices=prices,
            interval_hours=self.interval_hours,
            export_tariff=self.export_tariff,
        )

        scheduled_data = schedule_time_shiftable_loads(
            day_data=day_data,
            load_columns=load_columns,
            interval_hours=self.interval_hours,
            export_tariff=self.export_tariff,
        )

        optimized_grid = pd.to_numeric(
            scheduled_data["grid"],
            errors="coerce",
        )

        optimized_cost = calculate_net_energy_cost(
            grid_power=optimized_grid,
            prices=prices,
            interval_hours=self.interval_hours,
            export_tariff=self.export_tariff,
        )

        return HEMSDecision(
            agent_name=self.agent_name,
            original_cost=original_cost,
            optimized_cost=optimized_cost,
            cost_saving=original_cost - optimized_cost,
            scheduled_data=scheduled_data,
        )