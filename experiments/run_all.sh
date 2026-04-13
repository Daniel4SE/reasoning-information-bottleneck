#!/bin/bash
# Master experiment runner
# Usage: bash experiments/run_all.sh

set -e

echo "============================================"
echo "  RIG Experiment Pipeline"
echo "============================================"

# Step 1: Collect reasoning traces
# (Requires MLX or Transformers + model weights)
echo ""
echo "Step 1: Collecting reasoning traces..."
echo "  To collect real traces, run:"
echo "  python experiments/collect_reasoning_traces.py \\"
echo "    --model mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit \\"
echo "    --dataset gsm8k --max-samples 200 --max-tokens 2048 \\"
echo "    --backend mlx --output data/traces_deepseek_gsm8k.jsonl"
echo ""
echo "  Repeat for math, arc datasets."
echo ""

# Step 2: Analyze RIG
echo "Step 2: Analyzing RIG metrics..."
if [ -f "data/traces_deepseek_gsm8k.jsonl" ]; then
    python3 experiments/analyze_rig.py \
        --input data/traces_deepseek_gsm8k.jsonl \
        --output results/
else
    echo "  [SKIP] No trace files found. Run Step 1 first."
fi

# Step 3: Early stopping evaluation
echo ""
echo "Step 3: Evaluating early stopping..."
if [ -f "data/traces_deepseek_gsm8k.jsonl" ]; then
    python3 experiments/early_stopping.py \
        --input data/traces_deepseek_gsm8k.jsonl \
        --dataset gsm8k \
        --output results/
else
    echo "  [SKIP] No trace files found. Run Step 1 first."
fi

# Step 4: Generate figures
echo ""
echo "Step 4: Generating figures..."
python3 experiments/generate_figures.py \
    --results-dir results/ \
    --output-dir figures/

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Figures saved to figures/"
echo "============================================"
