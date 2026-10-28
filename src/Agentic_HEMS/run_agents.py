from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Agentic_HEMS.hems_agent import RuleBasedHEMSAgent


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_directory = project_root / "data" / "processed"
    results_directory = project_root / "results"
    results_directory.mkdir(parents=True, exist_ok=True)

    household_files = (
        sorted(
            data_directory.glob(
                "energy_data_prosumer_*.csv"
            )
        )
        + sorted(
            data_directory.glob(
                "energy_data_consumer_*.csv"
            )
        )
    )

    summary_records: list[dict] = []

    for index, csv_path in enumerate(household_files):
        agent_name = f"agent_{index}"
        json_path = csv_path.with_suffix(".json")

        household_data = pd.read_csv(csv_path).iloc[:96].copy()

        with json_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        agent = RuleBasedHEMSAgent(
            agent_name=agent_name,
            metadata=metadata,
            interval_hours=0.25,
            export_tariff=0.0097,
        )

        decision = agent.make_decision(household_data)

        summary_records.append(
            {
                "agent": agent_name,
                "participant_type": metadata[
                    "participant_type"
                ],
                "original_cost": decision.original_cost,
                "optimized_cost": decision.optimized_cost,
                "cost_saving": decision.cost_saving,
            }
        )

        schedule_path = (
            results_directory
            / f"{agent_name}_schedule.csv"
        )
        decision.scheduled_data.to_csv(
            schedule_path,
            index=False,
        )

    summary = pd.DataFrame(summary_records)

    summary_path = (
        results_directory / "multi_agent_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    print("\nHousehold agent decisions:\n")
    print(summary.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()