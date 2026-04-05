"""Public package exports for the trading workflow project."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import RunMode, Settings

if TYPE_CHECKING:
    from .workflow import TradingWorkflow

__all__ = ["RunMode", "Settings", "TradingWorkflow"]


def __getattr__(name: str):
    if name == "TradingWorkflow":
        from .workflow import TradingWorkflow

        return TradingWorkflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
