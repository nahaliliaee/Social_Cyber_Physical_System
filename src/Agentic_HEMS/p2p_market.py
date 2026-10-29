from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class P2PTrade:
    """One bilateral peer-to-peer energy transaction."""

    datetime: pd.Timestamp
    seller: str
    buyer: str
    energy_kwh: float
    price_per_kwh: float
    payment: float


class P2PEnergyMarket:
    """Baseline market for bilateral household energy trading."""

    def __init__(
        self,
        export_tariff: float = 0.0097,
        interval_hours: float = 0.25,
    ) -> None:
        self.export_tariff = export_tariff
        self.interval_hours = interval_hours

    def clear_market(
        self,
        timestamp: pd.Timestamp,
        grid_power_kw: dict[str, float],
        retail_prices: dict[str, float],
    ) -> list[P2PTrade]:
        """
        Match surplus households with deficit households.

        Negative grid power means export/surplus.
        Positive grid power means import/deficit.
        """
        sellers = [
            [agent, -power_kw * self.interval_hours]
            for agent, power_kw in sorted(
                grid_power_kw.items()
            )
            if power_kw < 0.0
        ]

        buyers = [
            [agent, power_kw * self.interval_hours]
            for agent, power_kw in sorted(
                grid_power_kw.items()
            )
            if power_kw > 0.0
        ]

        trades: list[P2PTrade] = []
        seller_index = 0
        buyer_index = 0
        tolerance = 1e-12

        while (
            seller_index < len(sellers)
            and buyer_index < len(buyers)
        ):
            seller, available_energy = sellers[seller_index]
            buyer, required_energy = buyers[buyer_index]

            retail_price = float(retail_prices[buyer])

            # Trading only occurs when both parties can benefit.
            if retail_price <= self.export_tariff:
                buyer_index += 1
                continue

            p2p_price = (
                retail_price + self.export_tariff
            ) / 2.0

            traded_energy = min(
                available_energy,
                required_energy,
            )

            if traded_energy > tolerance:
                trades.append(
                    P2PTrade(
                        datetime=timestamp,
                        seller=seller,
                        buyer=buyer,
                        energy_kwh=traded_energy,
                        price_per_kwh=p2p_price,
                        payment=traded_energy * p2p_price,
                    )
                )

            sellers[seller_index][1] -= traded_energy
            buyers[buyer_index][1] -= traded_energy

            if sellers[seller_index][1] <= tolerance:
                seller_index += 1

            if buyers[buyer_index][1] <= tolerance:
                buyer_index += 1

        return trades