#!/bin/bash
# ============================================================================
# DPG-Bench Inference with FlashU Acceleration
#
# Acceleration techniques (Paper Table 7, 1.5B configuration):
#   1. Task-Specific FFN Pruning (r_p=0.20)
#   2. Dynamic Layer Skipping (r_LS=0.20, T_LS=10)
#   3. Hybrid FFN Network (tau~0.3, last 10/34 steps use full network)
#   4. Adaptive Guidance Scale s(t) with 34-step schedule
#   5. Diffusion Head Cache (T_cache=5)
#
# Usage:
#   ./scripts/run_dpg_accelerated.sh           # Full run (all prompts)
#   ./scripts/run_dpg_accelerated.sh --demo    # Demo run (5 prompts)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

NUM_PROMPTS=1065
if [ "$1" = "--demo" ]; then
    NUM_PROMPTS=5
    echo "Running in DEMO mode (5 prompts only)"
fi

CUDA_VISIBLE_DEVICES=${GPU_ID:-0} python3 -m evaluation.inference_dpg_accelerated \
    --device_id 0 \
    --num_devices 1 \
    --log_dir logs_flashu \
    config=configs/showo2_1.5b_demo_432x432.yaml \
    outdir=output_flashu_dpg \
    validation_prompts_file=prompts/dpg_bench_meta_data.json \
    batch_size=1 \
    guidance_scale=10 \
    num_inference_steps=50 \
    num_prompts=$NUM_PROMPTS
