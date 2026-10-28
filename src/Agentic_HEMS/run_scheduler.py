from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Agentic_HEMS.shiftable_scheduler import (
    schedule_time_shiftable_loads,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_directory = project_root / "data" / "processed"
    results_directory = project_root / "results"
    results_directory.mkdir(parents=True, exist_ok=True)

    csv_path = data_directory / "energy_data_prosumer_661.csv"
    json_path = data_directory / "energy_data_prosumer_661.json"

    data = pd.read_csv(csv_path).iloc[:96].copy()

    with json_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    load_columns = metadata["time_shiftable_loads"]

    scheduled_data = schedule_time_shiftable_loads(
        day_data=data,
        load_columns=load_columns,
    )

    print("Time-shiftable load scheduling results:\n")

    for column in load_columns:
        original_energy = (
            pd.to_numeric(data[column], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
            .sum()
            * 0.25
        )

        scheduled_energy = (
            scheduled_data[column].sum() * 0.25
        )

        energy_difference = scheduled_energy - original_energy
        prices = pd.to_numeric(
            data["ToU"],
            errors="coerce",
        ).fillna(0.0)

        original_power = (
            pd.to_numeric(data[column], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )

        original_cost = float(
            (original_power * prices * 0.25).sum()
        )
        scheduled_cost = float(
            (scheduled_data[column] * prices * 0.25).sum()
        )
        cost_saving = original_cost - scheduled_cost
        print(f"{column}:")
        print(f"  Original energy:  {original_energy:.6f} kWh")
        print(f"  Scheduled energy: {scheduled_energy:.6f} kWh")
        print(f"  Difference:       {energy_difference:.10f} kWh")
        print(f"  Original cost:    ${original_cost:.6f}")
        print(f"  Scheduled cost:   ${scheduled_cost:.6f}")
        print(f"  Cost saving:      ${cost_saving:.6f}")
    output_path = (
        results_directory / "shiftable_schedule_test.csv"
    )
    scheduled_data.to_csv(output_path, index=False)

    print(f"\nScheduled data saved to: {output_path}")


if __name__ == "__main__":
    main()