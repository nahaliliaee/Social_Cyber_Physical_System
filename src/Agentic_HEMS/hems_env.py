from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


class HEMSEnvironment:
    """Baseline multi-household Home Energy Management environment."""

    STEPS_PER_DAY = 96
    INTERVAL_HOURS = 0.25

    def __init__(
        self,
        data_directory: str | Path,
        num_prosumers: int = 2,
        num_consumers: int = 2,
        experiment_days: int = 30,
        start_date: str | None = None,
        export_tariff: float = 0.0097,
        random_seed: int | None = None,
    ):
        self.data_directory = Path(data_directory)
        self.experiment_days = experiment_days
        self.experiment_steps = experiment_days * self.STEPS_PER_DAY
        self.start_date = start_date
        self.export_tariff = export_tariff
        self.random = random.Random(random_seed)

        self.agent_datasets: dict[str, pd.DataFrame] = {}
        self.agent_metadata: dict[str, dict] = {}

        self._load_households(
            num_prosumers=num_prosumers,
            num_consumers=num_consumers,
        )

        if not self.agent_datasets:
            raise ValueError(
                f"No processed household CSV files found in {self.data_directory}"
            )

        self.agents = list(self.agent_datasets)
        self.data_length = min(
            len(dataset) for dataset in self.agent_datasets.values()
        )

        if self.experiment_steps > self.data_length:
            raise ValueError(
                "experiment_days is longer than the available dataset."
            )

        self.current_index = 0
        self.step_number = 0

    def _load_households(
        self,
        num_prosumers: int,
        num_consumers: int,
    ) -> None:
        selected_files = (
            sorted(self.data_directory.glob("energy_data_prosumer_*.csv"))[
                :num_prosumers
            ]
            + sorted(self.data_directory.glob("energy_data_consumer_*.csv"))[
                :num_consumers
            ]
        )

        for index, csv_path in enumerate(selected_files):
            dataframe = pd.read_csv(csv_path)

            required_columns = {
                "datetime",
                "grid",
                "solar",
                "nonshiftable_loads",
                "day_of_week",
                "month",
                "Is_a_holiday",
                "ToU",
            }

            missing_columns = required_columns.difference(dataframe.columns)

            if missing_columns:
                raise ValueError(
                    f"{csv_path.name} is missing columns: "
                    f"{sorted(missing_columns)}"
                )

            dataframe["datetime"] = pd.to_datetime(
                dataframe["datetime"],
                format="mixed",
                utc=True,
            )
            dataframe = dataframe.sort_values("datetime").reset_index(drop=True)

            json_path = csv_path.with_suffix(".json")

            if json_path.exists():
                with json_path.open("r", encoding="utf-8") as file:
                    metadata = json.load(file)
            else:
                metadata = {
                    "participant_type": (
                        "prosumer"
                        if "prosumer" in csv_path.name
                        else "consumer"
                    ),
                    "power_shiftable_loads": [],
                    "time_shiftable_loads": [],
                }

            agent_name = f"agent_{index}"
            self.agent_datasets[agent_name] = dataframe
            self.agent_metadata[agent_name] = metadata

    def reset(self) -> dict[str, dict]:
        maximum_start = self.data_length - self.experiment_steps

        if self.start_date is not None:
            requested_date = pd.Timestamp(self.start_date).date()
            reference_data = next(iter(self.agent_datasets.values()))

            matching_rows = reference_data.index[
                reference_data["datetime"].dt.date == requested_date
            ].tolist()

            if not matching_rows:
                raise ValueError(
                    f"Start date {self.start_date} is not available."
                )

            self.current_index = matching_rows[0]

            if self.current_index > maximum_start:
                raise ValueError(
                    "Not enough data remains after the selected start date."
                )
        else:
            self.current_index = self.random.randint(0, maximum_start)

        self.step_number = 0
        return self.observe()

    def observe(self) -> dict[str, dict]:
        observations: dict[str, dict] = {}

        for agent, dataframe in self.agent_datasets.items():
            row = dataframe.iloc[self.current_index]
            metadata = self.agent_metadata[agent]

            shiftable_columns = (
                metadata.get("power_shiftable_loads", [])
                + metadata.get("time_shiftable_loads", [])
            )

            available_shiftable_columns = [
                column
                for column in shiftable_columns
                if column in dataframe.columns
            ]

            shiftable_load = float(
                row[available_shiftable_columns].sum()
                if available_shiftable_columns
                else 0.0
            )

            nonshiftable_load = float(row["nonshiftable_loads"])
            total_load = nonshiftable_load + shiftable_load

            observations[agent] = {
                "datetime": row["datetime"],
                "participant_type": metadata["participant_type"],
                "grid_kw": float(row["grid"]),
                "solar_kw": float(row["solar"]),
                "nonshiftable_load_kw": nonshiftable_load,
                "shiftable_load_kw": shiftable_load,
                "total_load_kw": total_load,
                "day_of_week": int(row["day_of_week"]),
                "month": int(row["month"]),
                "is_holiday": int(row["Is_a_holiday"]),
                "tou_price": float(row["ToU"]),
            }

        return observations

    def step(self) -> tuple[
        dict[str, dict],
        dict[str, float],
        dict[str, dict],
        bool,
    ]:
        observations = self.observe()
        rewards = self._calculate_rewards(observations)

        self.current_index += 1
        self.step_number += 1

        done = (
            self.step_number >= self.experiment_steps
            or self.current_index >= self.data_length
        )

        next_observations = {} if done else self.observe()

        return observations, rewards, next_observations, done

    def _calculate_rewards(
        self,
        observations: dict[str, dict],
    ) -> dict[str, float]:
        rewards: dict[str, float] = {}

        for agent, observation in observations.items():
            grid_kw = observation["grid_kw"]
            tou_price = observation["tou_price"]

            imported_energy_kwh = max(grid_kw, 0.0) * self.INTERVAL_HOURS
            exported_energy_kwh = max(-grid_kw, 0.0) * self.INTERVAL_HOURS

            energy_cost = imported_energy_kwh * tou_price
            export_revenue = exported_energy_kwh * self.export_tariff

            rewards[agent] = export_revenue - energy_cost

        return rewards