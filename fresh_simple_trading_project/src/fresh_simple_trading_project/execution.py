"""Order execution helpers and in-memory broker implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .data_collection import SimulatedAccountClient
from .models import AccountState, Action, Decision, ExecutionResult, RiskResult


class BrokerClient(Protocol):
    """Protocol for broker clients that can place orders."""

    def place_order(self, symbol: str, qty: int, side: str) -> str:
        ...


@dataclass
class InMemoryBrokerClient:
    """Broker stub that records submitted orders in memory."""

    submitted_orders: list[dict[str, str | int | float | None]] = field(default_factory=list)
    protective_orders: list[dict[str, str | int | float | None]] = field(default_factory=list)
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

    def sync_risk_orders(
        self,
        symbol: str,
        qty: int,
        *,
        stop_loss_price: float | None,
        take_profit_price: float | None,
    ) -> list[str]:
        self.clear_risk_orders(symbol)
        normalized_symbol = symbol.upper()
        order_ids: list[str] = []
        if qty <= 0:
            return order_ids
        if stop_loss_price is not None:
            order_id = f"paper-risk-stop-{len(self.protective_orders) + 1}"
            self.protective_orders.append(
                {
                    "symbol": normalized_symbol,
                    "qty": qty,
                    "side": "SELL",
                    "type": "stop",
                    "price": round(float(stop_loss_price), 2),
                    "order_id": order_id,
                }
            )
            order_ids.append(order_id)
        if take_profit_price is not None:
            order_id = f"paper-risk-tp-{len(self.protective_orders) + 1}"
            self.protective_orders.append(
                {
                    "symbol": normalized_symbol,
                    "qty": qty,
                    "side": "SELL",
                    "type": "limit",
                    "price": round(float(take_profit_price), 2),
                    "order_id": order_id,
                }
            )
            order_ids.append(order_id)
        return order_ids

    def clear_risk_orders(self, symbol: str) -> None:
        normalized_symbol = symbol.upper()
        self.protective_orders = [
            order for order in self.protective_orders if order.get("symbol") != normalized_symbol
        ]


@dataclass
class ExecutionModule:
    """Translate workflow decisions into broker operations."""

    broker_client: BrokerClient

    def execute(
        self,
        decision: Decision,
        execute_orders: bool,
        *,
        account: AccountState | None = None,
        risk: RiskResult | None = None,
    ) -> ExecutionResult:
        primary_order_id: str | None = None
        executed = False
        if decision.action != Action.HOLD and decision.quantity > 0:
            if not execute_orders:
                return ExecutionResult(executed=False, order_id=None, status="dry_run")
            primary_order_id = self.broker_client.place_order(
                symbol=decision.symbol,
                qty=decision.quantity,
                side=decision.action.value,
            )
            executed = True

        protective_order_ids = self._sync_risk_orders(
            decision=decision,
            execute_orders=execute_orders,
            account=account,
            risk=risk,
        )
        if executed:
            return ExecutionResult(
                executed=True,
                order_id=primary_order_id,
                status="submitted",
                protective_order_ids=protective_order_ids,
            )
        if protective_order_ids:
            return ExecutionResult(
                executed=False,
                order_id=None,
                status="protected",
                protective_order_ids=protective_order_ids,
            )
        return ExecutionResult(executed=False, order_id=None, status="skipped")

    def _sync_risk_orders(
        self,
        *,
        decision: Decision,
        execute_orders: bool,
        account: AccountState | None,
        risk: RiskResult | None,
    ) -> list[str]:
        if not execute_orders or account is None:
            return []
        managed_qty = int(account.position_qty)
        if decision.action == Action.BUY and decision.quantity > 0:
            managed_qty += int(decision.quantity)
        elif decision.action == Action.SELL and decision.quantity > 0:
            managed_qty = max(0, managed_qty - int(decision.quantity))

        if managed_qty <= 0 or risk is None or (risk.stop_loss_price is None and risk.take_profit_price is None):
            clearer = getattr(self.broker_client, "clear_risk_orders", None)
            if callable(clearer):
                clearer(decision.symbol)
            return []

        sync = getattr(self.broker_client, "sync_risk_orders", None)
        if not callable(sync):
            return []
        synced = sync(
            decision.symbol,
            managed_qty,
            stop_loss_price=risk.stop_loss_price,
            take_profit_price=risk.take_profit_price,
        )
        return [str(order_id) for order_id in synced]
