#!/usr/bin/env python3
"""
Evaluate information-guided early stopping on collected traces.

Simulates three early stopping strategies:
1. Fixed truncation (baseline): Cut at fixed percentage
2. Entropy threshold (baseline): Stop when token entropy drops below threshold
3. RIG-guided (ours): Stop when average RIG drops below δ × initial RIG

For each strategy, measures:
- Accuracy (by checking if truncated chain still contains the answer)
- Token savings percentage

Usage:
    python early_stopping.py --input data/traces_deepseek_gsm8k.jsonl --output results/
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np


def load_traces(path: str) -> list:
    """Load reasoning traces from JSONL."""
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def extract_answer(text: str, dataset: str = "gsm8k") -> str:
    """Extract the final numeric or letter answer from reasoning text."""
    if dataset in ("gsm8k", "math"):
        # Look for boxed answer
        boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
        if boxed:
            return boxed[-1].strip()
        # Look for "answer is X" patterns
        patterns = [
            r"(?:the\s+)?answer\s+is\s*:?\s*\$?([0-9,.\-]+)",
            r"(?:=\s*)([0-9,.\-]+)\s*$",
            r"####\s*([0-9,.\-]+)",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).replace(",", "").strip()
        # Last number in text
        numbers = re.findall(r"[0-9]+\.?[0-9]*", text)
        if numbers:
            return numbers[-1]
    elif dataset == "arc":
        # Look for letter answer
        patterns = [
            r"(?:answer|correct)\s+(?:is|:)\s*\(?([A-D])\)?",
            r"\b([A-D])\b\s*$",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
    return ""


def check_answer(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer."""
    pred = predicted.strip().lower().replace(",", "").rstrip(".")
    gold_clean = gold.strip().lower().replace(",", "").rstrip(".")
    if pred == gold_clean:
        return True
    # Try numeric comparison
    try:
        return abs(float(pred) - float(gold_clean)) < 1e-6
    except ValueError:
        return False


def fixed_truncation(trace: dict, fraction: float, dataset: str) -> dict:
    """Baseline: truncate at a fixed fraction of the chain."""
    n = trace["num_tokens"]
    cut = max(1, int(n * fraction))
    # We need the generated text up to position `cut`
    # Since we don't have per-token text, estimate from full text
    full_text = trace["generated_text"]
    # Approximate: cut text at proportional position
    char_cut = max(1, int(len(full_text) * fraction))
    truncated_text = full_text[:char_cut]

    predicted = extract_answer(truncated_text, dataset)
    correct = check_answer(predicted, trace["answer"])

    return {
        "correct": correct,
        "tokens_used": cut,
        "token_savings": 1.0 - fraction,
    }


def entropy_threshold_stopping(trace: dict, threshold: float, dataset: str) -> dict:
    """Baseline: stop when token log-prob entropy is low (model is confident)."""
    logprobs = np.array(trace["token_logprobs"])
    n = len(logprobs)

    # Compute running negative entropy (confidence)
    # Higher -logprob = lower confidence = higher entropy
    window = 10
    if n < window:
        stop_at = n
    else:
        running_neg_lp = np.convolve(-logprobs, np.ones(window) / window, mode="valid")
        # Stop when average negative log-prob drops below threshold
        # (model becomes very confident / repetitive)
        below = np.where(running_neg_lp < threshold)[0]
        if len(below) > 0:
            stop_at = int(below[0]) + window
        else:
            stop_at = n

    fraction = stop_at / n
    full_text = trace["generated_text"]
    char_cut = max(1, int(len(full_text) * fraction))
    truncated_text = full_text[:char_cut]

    predicted = extract_answer(truncated_text, dataset)
    correct = check_answer(predicted, trace["answer"])

    return {
        "correct": correct,
        "tokens_used": stop_at,
        "token_savings": 1.0 - stop_at / n,
    }


def rig_guided_stopping(
    trace: dict, delta: float, warmup: int, window: int, dataset: str
) -> dict:
    """
    Our method: information-guided early stopping.

    Stop when avg RIG over window drops below δ × initial avg RIG.
    """
    kl_divs = np.array(trace["kl_divergences"])
    n = len(kl_divs)

    if n < warmup + window:
        stop_at = n
    else:
        # Compute initial RIG level (during warmup/accumulation phase)
        initial_rig = np.mean(kl_divs[warmup // 2 : warmup])
        if initial_rig <= 0:
            initial_rig = np.mean(kl_divs[kl_divs > 0]) if np.any(kl_divs > 0) else 1.0

        threshold = delta * initial_rig
        stop_at = n  # default: no early stop

        for t in range(warmup, n - window + 1):
            avg_rig = np.mean(kl_divs[t : t + window])
            if avg_rig < threshold:
                stop_at = t + window
                break

    fraction = stop_at / n
    full_text = trace["generated_text"]
    char_cut = max(1, int(len(full_text) * fraction))
    truncated_text = full_text[:char_cut]

    predicted = extract_answer(truncated_text, dataset)
    correct = check_answer(predicted, trace["answer"])

    return {
        "correct": correct,
        "tokens_used": stop_at,
        "token_savings": 1.0 - stop_at / n,
    }


def evaluate_method(traces, method_fn, dataset, **kwargs):
    """Evaluate a stopping method across all traces."""
    results = []
    for trace in traces:
        r = method_fn(trace, dataset=dataset, **kwargs)
        results.append(r)

    accuracy = np.mean([r["correct"] for r in results])
    avg_savings = np.mean([r["token_savings"] for r in results])
    avg_tokens = np.mean([r["tokens_used"] for r in results])

    return {
        "accuracy": float(accuracy),
        "avg_token_savings": float(avg_savings),
        "avg_tokens_used": float(avg_tokens),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate early stopping methods")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    traces = load_traces(args.input)
    print(f"Loaded {len(traces)} traces for early stopping evaluation")

    # Full chain baseline
    full_results = []
    for trace in traces:
        predicted = extract_answer(trace["generated_text"], args.dataset)
        correct = check_answer(predicted, trace["answer"])
        full_results.append(correct)
    full_accuracy = np.mean(full_results)

    results = {
        "dataset": args.dataset,
        "n_traces": len(traces),
        "methods": {},
    }

    # 1. Full chain
    results["methods"]["full_chain"] = {
        "accuracy": float(full_accuracy),
        "avg_token_savings": 0.0,
        "avg_tokens_used": float(np.mean([t["num_tokens"] for t in traces])),
    }
    print(f"Full chain: accuracy={full_accuracy:.3f}")

    # 2. Fixed truncation baselines
    for frac in [0.3, 0.5, 0.7]:
        name = f"fixed_{int(frac * 100)}pct"
        r = evaluate_method(traces, fixed_truncation, args.dataset, fraction=frac)
        results["methods"][name] = r
        print(
            f"Fixed {int(frac * 100)}%: accuracy={r['accuracy']:.3f}, savings={r['avg_token_savings']:.3f}"
        )

    # 3. Entropy threshold baselines
    for thresh in [0.5, 1.0, 2.0]:
        name = f"entropy_thresh_{thresh}"
        r = evaluate_method(
            traces, entropy_threshold_stopping, args.dataset, threshold=thresh
        )
        results["methods"][name] = r
        print(
            f"Entropy θ={thresh}: accuracy={r['accuracy']:.3f}, savings={r['avg_token_savings']:.3f}"
        )

    # 4. RIG-guided (ours) with different δ values
    for delta in [0.05, 0.10, 0.15, 0.20]:
        name = f"rig_guided_delta_{delta}"
        r = evaluate_method(
            traces,
            rig_guided_stopping,
            args.dataset,
            delta=delta,
            warmup=30,
            window=20,
        )
        results["methods"][name] = r
        print(
            f"RIG δ={delta}: accuracy={r['accuracy']:.3f}, savings={r['avg_token_savings']:.3f}"
        )

    # Save
    basename = Path(args.input).stem
    out_path = os.path.join(args.output, f"{basename}_early_stopping.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
