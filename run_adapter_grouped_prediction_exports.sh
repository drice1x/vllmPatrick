#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/patrick/vllmPatrickMonitoring}"

cd "$ROOT"

if [[ -f "venv/bin/activate" ]]; then
  # Use the project environment if present.
  # These scripts operate on existing processed tables and do not need model serving.
  source venv/bin/activate
fi

echo "[1/2] Recomputing adapter-grouped ALFWorld bad_action prediction tables"
python paper_analysis_pipeline.py --root "$ROOT"

echo "[2/2] Recomputing Qwen exploit-action adapter-grouped tables"
python exploit_rescue_analysis.py --root "$ROOT"

echo
echo "[DONE] Wrote:"
echo "  $ROOT/tables/alfworld_prediction_by_ttc_adapter_featuregroup.csv"
echo "  $ROOT/tables_rescue/qwen_exploit_prediction_by_ttc_adapter_featuregroup.csv"
