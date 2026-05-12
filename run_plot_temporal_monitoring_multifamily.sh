#!/usr/bin/env bash
set -euo pipefail

cd /home/patrick/vllmPatrickMonitoring

python scripts/plot_temporal_monitoring_multifamily.py \
  --target_label exploit_action_t1 \
  --environments alfworld \
  --families Llama Qwen Falcon \
  --adapters mix05 mix10 mix50 \
  --output_dir /home/patrick/vllmPatrickMonitoring/figures/temporal \
  --min_count 2 \
  --n_bins 50 \
  --late_window_start 0.8 \
  --include_unconstrained
