from pathlib import Path

import pandas as pd

from Agentic_HEMS.hems_env import HEMSEnvironment


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    processed_data_directory = project_root / "data" / "processed"
    results_directory = project_root / "results"
    results_directory.mkdir(parents=True, exist_ok=True)

    environment = HEMSEnvironment(
        data_directory=processed_data_directory,
        num_prosumers=2,
        num_consumers=2,
        experiment_days=1,
        start_date=None,
        export_tariff=0.0097,
        random_seed=42,
    )

    observations = environment.reset()
    history: list[dict] = []
    done = False

    print("HEMS environment created successfully.")
    print(f"Agents: {environment.agents}")
    print(f"Simulation steps: {environment.experiment_steps}")

    while not done:
        observations, rewards, next_observations, done = environment.step()

        for agent, observation in observations.items():
            history.append(
                {
                    "step": environment.step_number - 1,
                    "agent": agent,
                    "datetime": observation["datetime"],
                    "participant_type": observation["participant_type"],
                    "grid_kw": observation["grid_kw"],
                    "solar_kw": observation["solar_kw"],
                    "nonshiftable_load_kw": observation[
                        "nonshiftable_load_kw"
                    ],
                    "shiftable_load_kw": observation["shiftable_load_kw"],
                    "total_load_kw": observation["total_load_kw"],
                    "tou_price": observation["tou_price"],
                    "reward": rewards[agent],
                }
            )

        observations = next_observations

    history_dataframe = pd.DataFrame(history)

    history_path = results_directory / "hems_baseline_history.csv"
    history_dataframe.to_csv(history_path, index=False)

    summary = (
        history_dataframe.groupby(
            ["agent", "participant_type"],
            as_index=False,
        )
        .agg(
            total_reward=("reward", "sum"),
            average_grid_kw=("grid_kw", "mean"),
            total_solar_kwh=("solar_kw", lambda values: values.sum() * 0.25),
            total_shiftable_kwh=(
                "shiftable_load_kw",
                lambda values: values.sum() * 0.25,
            ),
        )
    )

    summary_path = results_directory / "hems_baseline_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSimulation completed successfully.")
    print(f"History saved to: {history_path}")
    print(f"Summary saved to: {summary_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()