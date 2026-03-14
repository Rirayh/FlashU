#!/bin/bash
# ============================================================================
# DPG-Bench Baseline Inference (No Acceleration)
# Uses the original Show-o2 model without any pruning or acceleration.
#
# Usage:
#   ./scripts/run_dpg_baseline.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=${GPU_ID:-0} python3 -m evaluation.inference_dpg_baseline \
    config=configs/showo2_1.5b_demo_432x432.yaml \
    outdir=output_baseline_dpg \
    validation_prompts_file=prompts/dpg_bench_meta_data.json \
    batch_size=1 \
    guidance_scale=10 \
    num_inference_steps=50 \
    device_id=0 \
    num_devices=1
