"""Raw artifact store implementations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..models import NewsArticle
from ..utils import timestamp_slug
from .protocols import StorageRef


class LocalRawStore:
    """Persist raw artifacts on the local filesystem."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bars(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> StorageRef:
        """Persist raw bars to a local CSV file."""
        target = self.root / "bars" / f"{symbol}_{timeframe}_{timestamp_slug()}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        bars.to_csv(target)
        return StorageRef(uri=target.resolve().as_uri(), kind="bars", content_type="text/csv")

    def save_news(self, symbol: str, articles: list[NewsArticle]) -> StorageRef:
        """Persist raw news articles to a local JSON file."""
        target = self.root / "news" / f"{symbol}_{timestamp_slug()}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [article.__dict__ for article in articles]
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return StorageRef(uri=target.resolve().as_uri(), kind="news", content_type="application/json")


class AzureBlobRawStore:
    """Persist raw artifacts to Azure Blob Storage."""

    def __init__(
        self,
        *,
        container_name: str,
        account_url: str | None = None,
        connection_string: str | None = None,
        blob_prefix: str = "",
        blob_service_client_factory: Callable[["AzureBlobRawStore"], Any] | None = None,
    ) -> None:
        if not container_name.strip():
            raise RuntimeError("AzureBlobRawStore requires a non-empty container name.")
        if not (account_url or connection_string):
            raise RuntimeError(
                "AzureBlobRawStore requires AZURE_STORAGE_ACCOUNT_URL or AZURE_STORAGE_CONNECTION_STRING."
            )
        self.container_name = container_name.strip()
        self.account_url = account_url
        self.connection_string = connection_string
        self.blob_prefix = blob_prefix.strip("/")
        self._blob_service_client_factory = blob_service_client_factory
        self._blob_service_client: Any | None = None
        self._container_ready = False

    @property
    def blob_service_client(self) -> Any:
        """Return the lazily constructed Azure Blob service client."""
        if self._blob_service_client is None:
            factory = self._blob_service_client_factory or _default_blob_service_client_factory
            self._blob_service_client = factory(self)
        return self._blob_service_client

    def save_bars(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> StorageRef:
        """Persist raw bars to Azure Blob Storage."""
        payload = bars.to_csv()
        blob_name = self._dated_blob_name(
            "bars",
            symbol=symbol,
            subfolder=timeframe,
            suffix=".csv",
        )
        return self._upload_text(blob_name, payload, kind="bars", content_type="text/csv")

    def save_news(self, symbol: str, articles: list[NewsArticle]) -> StorageRef:
        """Persist raw news articles to Azure Blob Storage."""
        payload = json.dumps([article.__dict__ for article in articles], indent=2)
        blob_name = self._dated_blob_name(
            "news",
            symbol=symbol,
            suffix=".json",
        )
        return self._upload_text(blob_name, payload, kind="news", content_type="application/json")

    def _dated_blob_name(
        self,
        category: str,
        *,
        symbol: str,
        suffix: str,
        subfolder: str | None = None,
    ) -> str:
        current_time = datetime.now(timezone.utc)
        parts = []
        if self.blob_prefix:
            parts.append(self.blob_prefix)
        parts.extend([category, symbol.upper()])
        if subfolder:
            parts.append(subfolder)
        parts.extend(
            [
                current_time.strftime("%Y"),
                current_time.strftime("%m"),
                current_time.strftime("%d"),
                f"{timestamp_slug()}{suffix}",
            ]
        )
        return "/".join(parts)

    def _upload_text(self, blob_name: str, payload: str, *, kind: str, content_type: str) -> StorageRef:
        container_client = self._get_container_client()
        blob_client = container_client.get_blob_client(blob_name)
        content_settings = _build_content_settings(content_type)
        blob_client.upload_blob(payload, overwrite=True, content_settings=content_settings)
        return StorageRef(
            uri=str(getattr(blob_client, "url", "")) or self._fallback_blob_uri(blob_name),
            kind=kind,
            content_type=content_type,
        )

    def _get_container_client(self) -> Any:
        container_client = self.blob_service_client.get_container_client(self.container_name)
        if not self._container_ready:
            try:
                container_client.create_container()
            except Exception as exc:  # pragma: no cover - exercised with Azure SDK.
                if exc.__class__.__name__ != "ResourceExistsError":
                    raise
            self._container_ready = True
        return container_client

    def _fallback_blob_uri(self, blob_name: str) -> str:
        if self.account_url:
            return f"{self.account_url.rstrip('/')}/{self.container_name}/{blob_name}"
        return f"az://{self.container_name}/{blob_name}"


def _default_blob_service_client_factory(store: AzureBlobRawStore) -> Any:
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    if store.connection_string:
        return BlobServiceClient.from_connection_string(store.connection_string)
    return BlobServiceClient(account_url=store.account_url, credential=DefaultAzureCredential())


def _build_content_settings(content_type: str) -> Any:
    try:
        from azure.storage.blob import ContentSettings
    except ModuleNotFoundError:
        return {"content_type": content_type}

    return ContentSettings(content_type=content_type)
