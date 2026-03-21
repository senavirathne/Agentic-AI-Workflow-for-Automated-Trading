#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/agentic-alpaca-trading}"

sudo apt update
sudo apt install -y python3-venv python3-pip

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "Bootstrap complete. Create .env before running the workflow."
