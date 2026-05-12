#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/patrick/vllmPatrickMonitoring}"

cd "$ROOT"
python paper_analysis_pipeline.py --root "$ROOT"
