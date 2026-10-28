from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd


def schedule_time_shiftable_loads(
    day_data: pd.DataFrame,
    load_columns: Sequence[str],
    interval_hours: float = 0.25,
) -> pd.DataFrame:
    """
    Move time-shiftable loads to the cheapest consecutive periods.

    The scheduler:
    - preserves each appliance's total daily energy;
    - keeps its original maximum power;
    - schedules it as one continuous operating cycle;
    - clips small negative measurements to zero.
    """
    required_columns = {"datetime", "ToU"}
    missing_columns = required_columns.difference(day_data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    scheduled_data = day_data.copy().reset_index(drop=True)
    prices = pd.to_numeric(
        scheduled_data["ToU"],
        errors="coerce",
    )

    if prices.isna().any():
        raise ValueError("The ToU column contains invalid values.")

    for column in load_columns:
        if column not in scheduled_data.columns:
            continue

        baseline_power = pd.to_numeric(
            scheduled_data[column],
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0)

        daily_energy_kwh = float(
            baseline_power.sum() * interval_hours
        )
        maximum_power_kw = float(baseline_power.max())

        scheduled_data[column] = 0.0

        if daily_energy_kwh <= 0.0 or maximum_power_kw <= 0.0:
            continue

        required_intervals = math.ceil(
            daily_energy_kwh
            / (maximum_power_kw * interval_hours)
        )
        required_intervals = min(
            required_intervals,
            len(scheduled_data),
        )

        candidate_costs: list[tuple[float, int]] = []

        for start_index in range(
            len(scheduled_data) - required_intervals + 1
        ):
            end_index = start_index + required_intervals
            operating_cost = float(
                prices.iloc[start_index:end_index].sum()
            )
            candidate_costs.append(
                (operating_cost, start_index)
            )

        _, best_start = min(candidate_costs)
        remaining_energy_kwh = daily_energy_kwh

        for index in range(
            best_start,
            best_start + required_intervals,
        ):
            scheduled_power_kw = min(
                maximum_power_kw,
                remaining_energy_kwh / interval_hours,
            )
            scheduled_data.at[index, column] = scheduled_power_kw
            remaining_energy_kwh -= (
                scheduled_power_kw * interval_hours
            )

    return scheduled_data