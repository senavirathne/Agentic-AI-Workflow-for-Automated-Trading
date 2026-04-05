#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/dist/function_app_vm_dispatch"
ZIP_PATH="$ROOT_DIR/dist/function_app_vm_dispatch.zip"

rm -rf "$OUT_DIR" "$ZIP_PATH"
mkdir -p "$OUT_DIR"

cp "$ROOT_DIR/function_app.py" "$OUT_DIR/function_app.py"
cp "$ROOT_DIR/host.json" "$OUT_DIR/host.json"
cp "$ROOT_DIR/function_app.requirements.txt" "$OUT_DIR/requirements.txt"
cp "$ROOT_DIR/function_app.funcignore" "$OUT_DIR/.funcignore"
cp "$ROOT_DIR/local.settings.json.example" "$OUT_DIR/local.settings.json.example"
cp "$ROOT_DIR/function_app.README.md" "$OUT_DIR/README.md"

(
  cd "$OUT_DIR"
  zip -qr "$ZIP_PATH" .
)

printf 'Created %s\n' "$OUT_DIR"
printf 'Created %s\n' "$ZIP_PATH"
