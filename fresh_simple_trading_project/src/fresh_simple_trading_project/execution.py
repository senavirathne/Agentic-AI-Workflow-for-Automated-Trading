from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .data_collection import SimulatedAccountClient
from .models import Action, Decision, ExecutionResult


class BrokerClient(Protocol):
    def place_order(self, symbol: str, qty: int, side: str) -> str:
        ...


@dataclass
class InMemoryBrokerClient:
    submitted_orders: list[dict[str, str | int | float | None]] = field(default_factory=list)
    account_client: SimulatedAccountClient | None = None
    latest_prices: dict[str, float] = field(default_factory=dict)
    latest_timestamps: dict[str, str | None] = field(default_factory=dict)

    def set_market_price(self, symbol: str, price: float, timestamp: object | None = None) -> None:
        normalized_symbol = symbol.upper()
        self.latest_prices[normalized_symbol] = float(price)
        self.latest_timestamps[normalized_symbol] = None if timestamp is None else str(timestamp)

    def place_order(self, symbol: str, qty: int, side: str) -> str:
        normalized_symbol = symbol.upper()
        normalized_side = side.upper()
        order_id = f"paper-{len(self.submitted_orders) + 1}"
        price = self.latest_prices.get(normalized_symbol)
        if self.account_client is not None:
            if price is None:
                raise ValueError(f"No market price set for {normalized_symbol}.")
            self.account_client.apply_fill(normalized_symbol, qty, normalized_side, price)
        self.submitted_orders.append(
            {
                "symbol": normalized_symbol,
                "qty": qty,
                "side": normalized_side,
                "order_id": order_id,
                "price": price,
                "timestamp": self.latest_timestamps.get(normalized_symbol),
            }
        )
        return order_id


@dataclass
class ExecutionModule:
    broker_client: BrokerClient

    def execute(self, decision: Decision, execute_orders: bool) -> ExecutionResult:
        if decision.action == Action.HOLD or decision.quantity <= 0:
            return ExecutionResult(executed=False, order_id=None, status="skipped")
        if not execute_orders:
            return ExecutionResult(executed=False, order_id=None, status="dry_run")
        order_id = self.broker_client.place_order(
            symbol=decision.symbol,
            qty=decision.quantity,
            side=decision.action.value,
        )
        return ExecutionResult(executed=True, order_id=order_id, status="submitted")
