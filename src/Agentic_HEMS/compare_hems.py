from __future__ import annotations

from pathlib import Path

import pandas as pd

from Agentic_HEMS.hems_env import HEMSEnvironment
from Agentic_HEMS.optimized_hems_env import (
    OptimizedHEMSEnvironment,
)


def run_environment(
    environment: HEMSEnvironment,
    environment_name: str,
) -> pd.DataFrame:
    environment.reset()
    records: list[dict] = []
    done = False

    while not done:
        observations, rewards, _, done = environment.step()

        for agent, observation in observations.items():
            records.append(
                {
                    "environment": environment_name,
                    "agent": agent,
                    "participant_type": observation[
                        "participant_type"
                    ],
                    "reward": rewards[agent],
                    "original_time_load_kw": observation.get(
                        "original_time_shiftable_load_kw",
                        0.0,
                    ),
                    "scheduled_time_load_kw": observation.get(
                        "scheduled_time_shiftable_load_kw",
                        0.0,
                    ),
                }
            )

    return pd.DataFrame(records)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_directory = project_root / "data" / "processed"
    results_directory = project_root / "results"
    results_directory.mkdir(parents=True, exist_ok=True)

    common_settings = {
        "data_directory": data_directory,
        "num_prosumers": 2,
        "num_consumers": 2,
        "experiment_days": 1,
        "start_date": None,
        "export_tariff": 0.0097,
        "random_seed": 42,
    }

    baseline_environment = HEMSEnvironment(
        **common_settings,
    )
    optimized_environment = OptimizedHEMSEnvironment(
        **common_settings,
    )

    baseline_results = run_environment(
        baseline_environment,
        "baseline",
    )
    optimized_results = run_environment(
        optimized_environment,
        "optimized",
    )

    baseline_summary = (
        baseline_results.groupby(
            ["agent", "participant_type"],
            as_index=False,
        )
        .agg(baseline_reward=("reward", "sum"))
    )

    optimized_summary = (
        optimized_results.groupby(
            ["agent", "participant_type"],
            as_index=False,
        )
        .agg(
            optimized_reward=("reward", "sum"),
            original_time_energy_kwh=(
                "original_time_load_kw",
                lambda values: values.sum() * 0.25,
            ),
            scheduled_time_energy_kwh=(
                "scheduled_time_load_kw",
                lambda values: values.sum() * 0.25,
            ),
        )
    )

    comparison = baseline_summary.merge(
        optimized_summary,
        on=["agent", "participant_type"],
    )

    comparison["baseline_net_cost"] = (
        -comparison["baseline_reward"]
    )
    comparison["optimized_net_cost"] = (
        -comparison["optimized_reward"]
    )
    comparison["cost_saving"] = (
        comparison["baseline_net_cost"]
        - comparison["optimized_net_cost"]
    )
    comparison["energy_difference_kwh"] = (
        comparison["scheduled_time_energy_kwh"]
        - comparison["original_time_energy_kwh"]
    )

    output_path = (
        results_directory / "hems_comparison.csv"
    )
    comparison.to_csv(output_path, index=False)

    display_columns = [
        "agent",
        "participant_type",
        "baseline_net_cost",
        "optimized_net_cost",
        "cost_saving",
        "energy_difference_kwh",
    ]

    print("\nBaseline versus optimized HEMS:\n")
    print(
        comparison[display_columns].to_string(index=False)
    )
    print(f"\nComparison saved to: {output_path}")


if __name__ == "__main__":
    main()