#!/usr/bin/env bash
set -euo pipefail

cd /home/patrick/vllmPatrickMonitoring

if [[ -f "venv/bin/activate" ]]; then
  source venv/bin/activate
fi

python scripts/render_qwen_bad_action_temporal_grid.py
