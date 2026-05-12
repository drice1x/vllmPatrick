#!/usr/bin/env bash
set -euo pipefail

cd /home/patrick/vllmPatrickMonitoring
python scripts/render_qwen_bad_action_all_ttc.py "$@"
