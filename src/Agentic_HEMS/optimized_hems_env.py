from __future__ import annotations

import pandas as pd

from Agentic_HEMS.hems_env import HEMSEnvironment
from Agentic_HEMS.shiftable_scheduler import (
    schedule_time_shiftable_loads,
)


class OptimizedHEMSEnvironment(HEMSEnvironment):
    """HEMS environment with scheduled time-shiftable loads."""

    def reset(self) -> dict[str, dict]:
        self._scheduled_data: dict[str, pd.DataFrame] = {}

        # Select the same simulation starting point as the baseline.
        super().reset()

        for agent, dataframe in self.agent_datasets.items():
            metadata = self.agent_metadata[agent]
            time_shiftable_columns = metadata.get(
                "time_shiftable_loads",
                [],
            )

            experiment_data = dataframe.iloc[
                self.current_index:
                self.current_index + self.experiment_steps
            ].copy()

            scheduled_days: list[pd.DataFrame] = []

            for day_start in range(
                0,
                len(experiment_data),
                self.STEPS_PER_DAY,
            ):
                day_data = experiment_data.iloc[
                    day_start:
                    day_start + self.STEPS_PER_DAY
                ].copy()

                scheduled_day = schedule_time_shiftable_loads(
                    day_data=day_data,
                    load_columns=time_shiftable_columns,
                    interval_hours=self.INTERVAL_HOURS,
                )
                scheduled_days.append(scheduled_day)

            self._scheduled_data[agent] = pd.concat(
                scheduled_days,
                ignore_index=True,
            )

        self.step_number = 0
        return self.observe()

    def observe(self) -> dict[str, dict]:
        observations = super().observe()

        if not getattr(self, "_scheduled_data", None):
            return observations

        for agent, observation in observations.items():
            scheduled_row = self._scheduled_data[agent].iloc[
                self.step_number
            ]
            original_row = self.agent_datasets[agent].iloc[
                self.current_index
            ]
            metadata = self.agent_metadata[agent]

            time_columns = [
                column
                for column in metadata.get(
                    "time_shiftable_loads",
                    [],
                )
                if column in original_row.index
            ]
            power_columns = [
                column
                for column in metadata.get(
                    "power_shiftable_loads",
                    [],
                )
                if column in original_row.index
            ]

            original_time_load = sum(
                max(float(original_row[column]), 0.0)
                for column in time_columns
            )
            scheduled_time_load = sum(
                max(float(scheduled_row[column]), 0.0)
                for column in time_columns
            )
            power_shiftable_load = sum(
                max(float(original_row[column]), 0.0)
                for column in power_columns
            )

            controlled_shiftable_load = (
                power_shiftable_load + scheduled_time_load
            )

            grid_adjustment = (
                scheduled_time_load - original_time_load
            )
            controlled_grid = (
                float(original_row["grid"]) + grid_adjustment
            )

            observation["original_time_shiftable_load_kw"] = (
                original_time_load
            )
            observation["scheduled_time_shiftable_load_kw"] = (
                scheduled_time_load
            )
            observation["shiftable_load_kw"] = (
                controlled_shiftable_load
            )
            observation["total_load_kw"] = (
                observation["nonshiftable_load_kw"]
                + controlled_shiftable_load
            )
            observation["grid_kw"] = controlled_grid

        return observations