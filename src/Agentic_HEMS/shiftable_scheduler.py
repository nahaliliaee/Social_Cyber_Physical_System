from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd


def calculate_net_energy_cost(
    grid_power: pd.Series,
    prices: pd.Series,
    interval_hours: float,
    export_tariff: float,
) -> float:
    """Calculate import cost minus export revenue."""
    imported_power = grid_power.clip(lower=0.0)
    exported_power = (-grid_power).clip(lower=0.0)

    import_cost = (
        imported_power * prices * interval_hours
    ).sum()
    export_revenue = (
        exported_power * export_tariff * interval_hours
    ).sum()

    return float(import_cost - export_revenue)


def schedule_time_shiftable_loads(
    day_data: pd.DataFrame,
    load_columns: Sequence[str],
    interval_hours: float = 0.25,
    export_tariff: float = 0.0097,
) -> pd.DataFrame:
    """
    Schedule time-shiftable loads using whole-house net cost.

    The scheduler:
    - preserves each appliance's total energy;
    - keeps its original maximum power;
    - tests continuous operating cycles;
    - considers grid imports and export revenue;
    - keeps the baseline schedule if optimization is worse.
    """
    required_columns = {"datetime", "ToU", "grid"}
    missing_columns = required_columns.difference(
        day_data.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    scheduled_data = day_data.copy().reset_index(drop=True)

    prices = pd.to_numeric(
        scheduled_data["ToU"],
        errors="coerce",
    )

    working_grid = pd.to_numeric(
        scheduled_data["grid"],
        errors="coerce",
    )

    if prices.isna().any():
        raise ValueError("The ToU column contains invalid values.")

    if working_grid.isna().any():
        raise ValueError("The grid column contains invalid values.")

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

        if daily_energy_kwh <= 0.0 or maximum_power_kw <= 0.0:
            scheduled_data[column] = 0.0
            continue

        grid_without_appliance = (
            working_grid - baseline_power
        )

        baseline_cost = calculate_net_energy_cost(
            grid_power=working_grid,
            prices=prices,
            interval_hours=interval_hours,
            export_tariff=export_tariff,
        )

        required_intervals = math.ceil(
            daily_energy_kwh
            / (maximum_power_kw * interval_hours)
        )
        required_intervals = min(
            required_intervals,
            len(scheduled_data),
        )

        best_profile = baseline_power.copy()
        best_cost = baseline_cost

        for start_index in range(
            len(scheduled_data) - required_intervals + 1
        ):
            candidate_profile = pd.Series(
                0.0,
                index=scheduled_data.index,
            )
            remaining_energy_kwh = daily_energy_kwh

            for index in range(
                start_index,
                start_index + required_intervals,
            ):
                scheduled_power_kw = min(
                    maximum_power_kw,
                    remaining_energy_kwh / interval_hours,
                )
                candidate_profile.iloc[index] = (
                    scheduled_power_kw
                )
                remaining_energy_kwh -= (
                    scheduled_power_kw * interval_hours
                )

            candidate_grid = (
                grid_without_appliance + candidate_profile
            )

            candidate_cost = calculate_net_energy_cost(
                grid_power=candidate_grid,
                prices=prices,
                interval_hours=interval_hours,
                export_tariff=export_tariff,
            )

            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_profile = candidate_profile

        scheduled_data[column] = best_profile
        working_grid = grid_without_appliance + best_profile

    scheduled_data["grid"] = working_grid

    return scheduled_data