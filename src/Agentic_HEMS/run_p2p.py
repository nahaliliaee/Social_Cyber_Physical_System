from __future__ import annotations

from pathlib import Path

import pandas as pd

from Agentic_HEMS.p2p_market import P2PEnergyMarket


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    results_directory = project_root / "results"
    results_directory.mkdir(parents=True, exist_ok=True)

    schedule_paths = sorted(
        results_directory.glob("agent_*_schedule.csv")
    )

    if not schedule_paths:
        raise FileNotFoundError(
            "No agent schedule files were found. "
            "Run Agentic_HEMS.run_agents first."
        )

    schedules = {
        path.stem.replace("_schedule", ""): pd.read_csv(path)
        for path in schedule_paths
    }

    number_of_steps = min(
        len(schedule) for schedule in schedules.values()
    )

    market = P2PEnergyMarket(
        export_tariff=0.0097,
        interval_hours=0.25,
    )

    trade_records: list[dict] = []

    for step in range(number_of_steps):
        first_schedule = next(iter(schedules.values()))
        timestamp = pd.Timestamp(
            first_schedule.iloc[step]["datetime"]
        )

        grid_power = {
            agent: float(schedule.iloc[step]["grid"])
            for agent, schedule in schedules.items()
        }

        retail_prices = {
            agent: float(schedule.iloc[step]["ToU"])
            for agent, schedule in schedules.items()
        }

        trades = market.clear_market(
            timestamp=timestamp,
            grid_power_kw=grid_power,
            retail_prices=retail_prices,
        )

        for trade in trades:
            buyer_retail_price = retail_prices[trade.buyer]

            buyer_saving = (
                buyer_retail_price - trade.price_per_kwh
            ) * trade.energy_kwh

            seller_gain = (
                trade.price_per_kwh - market.export_tariff
            ) * trade.energy_kwh

            trade_records.append(
                {
                    "datetime": trade.datetime,
                    "seller": trade.seller,
                    "buyer": trade.buyer,
                    "energy_kwh": trade.energy_kwh,
                    "price_per_kwh": trade.price_per_kwh,
                    "payment": trade.payment,
                    "buyer_saving": buyer_saving,
                    "seller_gain": seller_gain,
                    "community_benefit": (
                        buyer_saving + seller_gain
                    ),
                }
            )

    trades_dataframe = pd.DataFrame(trade_records)

    if trades_dataframe.empty:
        print("No P2P trades occurred.")
        return

    agent_records: list[dict] = []

    for agent in sorted(schedules):
        sold_trades = trades_dataframe[
            trades_dataframe["seller"] == agent
        ]
        bought_trades = trades_dataframe[
            trades_dataframe["buyer"] == agent
        ]

        agent_records.append(
            {
                "agent": agent,
                "energy_sold_kwh": sold_trades[
                    "energy_kwh"
                ].sum(),
                "energy_bought_kwh": bought_trades[
                    "energy_kwh"
                ].sum(),
                "seller_gain": sold_trades[
                    "seller_gain"
                ].sum(),
                "buyer_saving": bought_trades[
                    "buyer_saving"
                ].sum(),
                "total_benefit": (
                    sold_trades["seller_gain"].sum()
                    + bought_trades["buyer_saving"].sum()
                ),
            }
        )

    agent_summary = pd.DataFrame(agent_records)

    trades_path = results_directory / "p2p_trades.csv"
    summary_path = (
        results_directory / "p2p_agent_summary.csv"
    )

    trades_dataframe.to_csv(trades_path, index=False)
    agent_summary.to_csv(summary_path, index=False)

    print("\nP2P market completed successfully.")
    print(f"Number of trades: {len(trades_dataframe)}")
    print(
        "Total traded energy: "
        f"{trades_dataframe['energy_kwh'].sum():.6f} kWh"
    )
    print(
        "Total community benefit: $"
        f"{trades_dataframe['community_benefit'].sum():.6f}"
    )

    print("\nAgent settlement summary:\n")
    print(agent_summary.to_string(index=False))

    print(f"\nTrades saved to: {trades_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()