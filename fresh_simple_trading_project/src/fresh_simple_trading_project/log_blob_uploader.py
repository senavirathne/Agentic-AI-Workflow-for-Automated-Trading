"""Upload workflow log files to Azure Blob Storage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import Settings


def upload_log_file_to_blob(
    *,
    log_file_path: Path,
    container_name: str,
    blob_name: str,
    account_url: str | None = None,
    connection_string: str | None = None,
) -> str:
    """Upload one log file to Azure Blob Storage and return the blob URL.

    Args:
        log_file_path: Local workflow log file to upload.
        container_name: Blob container name to write into.
        blob_name: Destination blob path inside the container.
        account_url: Optional Azure Blob account URL.
        connection_string: Optional Azure Storage connection string.

    Returns:
        The uploaded blob URL.
    """
    if not log_file_path.is_file():
        raise RuntimeError(f"Workflow log file does not exist: {log_file_path}")

    resolved_account_url = account_url
    resolved_connection_string = connection_string
    if not (resolved_account_url or resolved_connection_string):
        settings = Settings.from_env(Path.cwd())
        resolved_account_url = settings.azure.storage_account_url
        resolved_connection_string = settings.azure.storage_connection_string

    if not (resolved_account_url or resolved_connection_string):
        raise RuntimeError(
            "Missing Azure Storage configuration for log upload. "
            "Set AZURE_STORAGE_ACCOUNT_URL or AZURE_STORAGE_CONNECTION_STRING on the VM."
        )

    service_client = _build_blob_service_client(
        account_url=resolved_account_url,
        connection_string=resolved_connection_string,
    )
    container_client = service_client.get_container_client(container_name)
    try:
        container_client.create_container()
    except Exception as exc:  # pragma: no cover - exercised with Azure SDK.
        if exc.__class__.__name__ != "ResourceExistsError":
            raise

    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(
        log_file_path.read_bytes(),
        overwrite=True,
        content_settings=_build_content_settings("text/plain; charset=utf-8"),
    )
    return str(getattr(blob_client, "url", "")).strip()


def main() -> None:
    """Upload a workflow log file to Azure Blob Storage from the CLI."""
    parser = argparse.ArgumentParser(description="Upload a workflow log file to Azure Blob Storage.")
    parser.add_argument("--file", required=True, dest="log_file_path")
    parser.add_argument("--container", required=True)
    parser.add_argument("--blob-name", required=True)
    parser.add_argument("--account-url", default=os.environ.get("VM_LOG_BLOB_ACCOUNT_URL"))
    parser.add_argument("--connection-string", default=os.environ.get("VM_LOG_BLOB_CONNECTION_STRING"))
    args = parser.parse_args()

    blob_url = upload_log_file_to_blob(
        log_file_path=Path(args.log_file_path).expanduser().resolve(),
        container_name=args.container,
        blob_name=args.blob_name,
        account_url=args.account_url,
        connection_string=args.connection_string,
    )
    print(blob_url)


def _build_blob_service_client(*, account_url: str | None, connection_string: str | None):
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    if connection_string:
        return BlobServiceClient.from_connection_string(connection_string)
    return BlobServiceClient(
        account_url=account_url,
        credential=DefaultAzureCredential(exclude_interactive_browser_credential=True),
    )


def _build_content_settings(content_type: str):
    from azure.storage.blob import ContentSettings

    return ContentSettings(content_type=content_type)


if __name__ == "__main__":
    main()
