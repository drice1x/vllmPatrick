#!/usr/bin/env bash
set -euo pipefail

cd /home/patrick/vllmPatrickMonitoring
source venv/bin/activate
python scripts/render_main_alfworld_prediction_ablation.py
