#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/patrick/vllmPatrickMonitoring}"

cd "$ROOT"
python final_agentic_evidence_pipeline.py --root "$ROOT"
