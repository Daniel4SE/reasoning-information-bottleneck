#!/usr/bin/env python3
"""
Analyze Reasoning Information Gain (RIG) from collected traces.

Computes:
1. Token-level RIG (from KL divergences)
2. Cumulative Reasoning Information (CRI) curves
3. Three-phase detection (accumulation, plateau, convergence)
4. Redundancy ratios
5. Theoretical lower bound validation

Input: JSONL files from collect_reasoning_traces.py
Output: JSON summary + numpy arrays for plotting

Usage:
    python analyze_rig.py --input data/traces_deepseek_gsm8k.jsonl --output results/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.stats import entropy


def load_traces(path: str) -> list:
    """Load reasoning traces from JSONL."""
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def compute_rig(kl_divergences: list) -> np.ndarray:
    """
    Compute Reasoning Information Gain from KL divergences.
    RIG(t) ≈ KL(p_t || p_{t-1}) - the shift in predictive distribution.
    """
    rig = np.array(kl_divergences, dtype=np.float64)
    # Clip extreme values (numerical artifacts)
    rig = np.clip(
        rig, 0, np.percentile(rig[rig > 0], 99) * 3 if np.any(rig > 0) else 1.0
    )
    return rig


def compute_cri(rig: np.ndarray) -> np.ndarray:
    """Compute Cumulative Reasoning Information."""
    return np.cumsum(rig)


def compute_efficiency(cri: np.ndarray) -> np.ndarray:
    """Compute reasoning efficiency η(t) = CRI(t) / CRI(T)."""
    total = cri[-1] if cri[-1] > 0 else 1.0
    return cri / total


def find_min_effective_length(efficiency: np.ndarray, alpha: float = 0.95) -> int:
    """Find T*(α) = min{t : η(t) ≥ α}."""
    indices = np.where(efficiency >= alpha)[0]
    if len(indices) > 0:
        return int(indices[0]) + 1  # 1-indexed
    return len(efficiency)


def detect_phases(rig: np.ndarray, window: int = 20) -> dict:
    """
    Detect three phases in the RIG sequence using changepoint detection.

    Phase 1 (Accumulation): High and increasing RIG
    Phase 2 (Plateau): Low, approximately constant RIG
    Phase 3 (Convergence): Possible spike then decay

    Uses smoothed RIG with rolling mean and gradient analysis.
    """
    n = len(rig)
    if n < 3 * window:
        return {"t1": n // 3, "t2": 2 * n // 3, "valid": False}

    # Smooth the RIG signal
    smoothed = np.convolve(rig, np.ones(window) / window, mode="same")

    # Compute gradient of smoothed RIG
    gradient = np.gradient(smoothed)

    # Phase 1 -> Phase 2 transition: first sustained drop in RIG
    # Find where smoothed RIG drops below 50% of its initial peak
    peak_val = np.max(smoothed[: n // 3]) if n > 0 else 0
    if peak_val > 0:
        below_half = np.where(smoothed[window:] < 0.3 * peak_val)[0]
        if len(below_half) > 0:
            t1 = int(below_half[0]) + window
        else:
            t1 = n // 3
    else:
        t1 = n // 3

    # Phase 2 -> Phase 3 transition: late spike or final 20% of chain
    # Look for increase in RIG in the last portion
    last_portion = smoothed[max(t1, n // 2) :]
    if len(last_portion) > window:
        late_gradient = np.gradient(last_portion)
        # Find significant increase in the tail
        increases = np.where(late_gradient > np.std(late_gradient))[0]
        if len(increases) > 0:
            t2 = int(increases[0]) + max(t1, n // 2)
        else:
            t2 = int(0.85 * n)
    else:
        t2 = int(0.85 * n)

    # Ensure ordering
    t1 = max(window, min(t1, n - 2 * window))
    t2 = max(t1 + window, min(t2, n - window))

    return {"t1": int(t1), "t2": int(t2), "valid": True}


def compute_entropy_rate(token_logprobs: list) -> float:
    """
    Estimate entropy rate h_r from token log-probabilities.
    h_r ≈ -1/T Σ log p(r_t | r_{<t})
    """
    logprobs = np.array(token_logprobs, dtype=np.float64)
    # Entropy rate = average negative log-prob
    return float(-np.mean(logprobs))


def compute_theoretical_bound(alpha: float, I_total: float, h_r: float) -> float:
    """
    Compute theoretical lower bound: T*(α) ≥ α * I_total / h_r
    """
    if h_r <= 0:
        return 0.0
    return alpha * I_total / h_r


def analyze_single_trace(trace: dict) -> dict:
    """Analyze a single reasoning trace."""
    kl_divs = trace["kl_divergences"]
    logprobs = trace["token_logprobs"]
    n_tokens = trace["num_tokens"]

    if n_tokens < 10:
        return None

    # Core computations
    rig = compute_rig(kl_divs)
    cri = compute_cri(rig)
    efficiency = compute_efficiency(cri)

    # Minimum effective lengths
    t_star_90 = find_min_effective_length(efficiency, 0.90)
    t_star_95 = find_min_effective_length(efficiency, 0.95)
    t_star_99 = find_min_effective_length(efficiency, 0.99)

    # Redundancy ratio
    rho = 1.0 - t_star_95 / n_tokens

    # Phase detection
    phases = detect_phases(rig)

    # Information in each phase
    if phases["valid"]:
        t1, t2 = phases["t1"], phases["t2"]
        phase1_info = float(cri[min(t1, n_tokens - 1)])
        phase2_info = float(cri[min(t2, n_tokens - 1)] - cri[min(t1, n_tokens - 1)])
        phase3_info = float(cri[-1] - cri[min(t2, n_tokens - 1)])
        total_info = float(cri[-1])

        phase1_frac = phase1_info / total_info if total_info > 0 else 0
        phase2_frac = phase2_info / total_info if total_info > 0 else 0
        phase3_frac = phase3_info / total_info if total_info > 0 else 0

        phase1_tokens_frac = t1 / n_tokens
        phase2_tokens_frac = (t2 - t1) / n_tokens
        phase3_tokens_frac = (n_tokens - t2) / n_tokens
    else:
        phase1_frac = phase2_frac = phase3_frac = 1 / 3
        phase1_tokens_frac = phase2_tokens_frac = phase3_tokens_frac = 1 / 3

    # Entropy rate
    h_r = compute_entropy_rate(logprobs)

    # Total reasoning information
    I_total = float(cri[-1])

    # Theoretical bound
    bound_95 = compute_theoretical_bound(0.95, I_total, h_r)

    return {
        "id": trace["id"],
        "question": trace["question"][:100],
        "difficulty": trace.get("difficulty", "unknown"),
        "n_tokens": n_tokens,
        "t_star_90": t_star_90,
        "t_star_95": t_star_95,
        "t_star_99": t_star_99,
        "redundancy_ratio": rho,
        "entropy_rate": h_r,
        "I_total": I_total,
        "theoretical_bound_95": bound_95,
        "bound_tightness": bound_95 / t_star_95 if t_star_95 > 0 else 0,
        "phases": phases,
        "phase1_info_frac": phase1_frac,
        "phase2_info_frac": phase2_frac,
        "phase3_info_frac": phase3_frac,
        "phase1_tokens_frac": phase1_tokens_frac,
        "phase2_tokens_frac": phase2_tokens_frac,
        "phase3_tokens_frac": phase3_tokens_frac,
        # Raw arrays for plotting
        "rig": rig.tolist(),
        "cri": cri.tolist(),
        "efficiency": efficiency.tolist(),
    }


def aggregate_results(analyses: list) -> dict:
    """Compute aggregate statistics across all traces."""
    valid = [a for a in analyses if a is not None]
    if not valid:
        return {}

    def safe_mean(key):
        vals = [a[key] for a in valid if key in a]
        return float(np.mean(vals)) if vals else 0.0

    def safe_std(key):
        vals = [a[key] for a in valid if key in a]
        return float(np.std(vals)) if vals else 0.0

    return {
        "n_traces": len(valid),
        "avg_tokens": safe_mean("n_tokens"),
        "avg_t_star_95": safe_mean("t_star_95"),
        "avg_redundancy": safe_mean("redundancy_ratio"),
        "std_redundancy": safe_std("redundancy_ratio"),
        "avg_entropy_rate": safe_mean("entropy_rate"),
        "avg_I_total": safe_mean("I_total"),
        "avg_theoretical_bound": safe_mean("theoretical_bound_95"),
        "avg_bound_tightness": safe_mean("bound_tightness"),
        "avg_phase1_info": safe_mean("phase1_info_frac"),
        "avg_phase2_info": safe_mean("phase2_info_frac"),
        "avg_phase3_info": safe_mean("phase3_info_frac"),
        "avg_phase1_tokens": safe_mean("phase1_tokens_frac"),
        "avg_phase2_tokens": safe_mean("phase2_tokens_frac"),
        "avg_phase3_tokens": safe_mean("phase3_tokens_frac"),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze RIG from collected traces")
    parser.add_argument(
        "--input", type=str, required=True, help="Input JSONL trace file"
    )
    parser.add_argument(
        "--output", type=str, default="results/", help="Output directory"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading traces from {args.input}...")
    traces = load_traces(args.input)
    print(f"Loaded {len(traces)} traces")

    # Analyze each trace
    analyses = []
    for i, trace in enumerate(traces):
        if (i + 1) % 50 == 0:
            print(f"  Analyzing {i + 1}/{len(traces)}...")
        result = analyze_single_trace(trace)
        if result is not None:
            analyses.append(result)

    # Aggregate
    agg = aggregate_results(analyses)

    # Save results
    basename = Path(args.input).stem
    detail_path = os.path.join(args.output, f"{basename}_analysis.jsonl")
    summary_path = os.path.join(args.output, f"{basename}_summary.json")

    with open(detail_path, "w") as f:
        for a in analyses:
            # Don't save large arrays in detailed JSONL (save separately)
            compact = {
                k: v for k, v in a.items() if k not in ("rig", "cri", "efficiency")
            }
            f.write(json.dumps(compact) + "\n")

    # Save CRI curves as numpy for plotting
    cri_curves = [np.array(a["cri"]) for a in analyses]
    rig_curves = [np.array(a["rig"]) for a in analyses]
    np.savez(
        os.path.join(args.output, f"{basename}_curves.npz"),
        cri_curves=np.array(cri_curves, dtype=object),
        rig_curves=np.array(rig_curves, dtype=object),
    )

    with open(summary_path, "w") as f:
        json.dump(agg, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Traces analyzed: {agg.get('n_traces', 0)}")
    print(f"Avg reasoning length: {agg.get('avg_tokens', 0):.0f} tokens")
    print(f"Avg T*(0.95): {agg.get('avg_t_star_95', 0):.0f} tokens")
    print(f"Avg redundancy ratio: {agg.get('avg_redundancy', 0):.3f}")
    print(f"Avg entropy rate: {agg.get('avg_entropy_rate', 0):.3f} nats/token")
    print(f"Avg theoretical bound: {agg.get('avg_theoretical_bound', 0):.0f} tokens")
    print(f"Bound tightness: {agg.get('avg_bound_tightness', 0):.3f}")
    print()
    print("Phase analysis:")
    print(
        f"  Phase 1 (Accumulation): {agg.get('avg_phase1_tokens', 0) * 100:.1f}% tokens, "
        f"{agg.get('avg_phase1_info', 0) * 100:.1f}% info"
    )
    print(
        f"  Phase 2 (Plateau):      {agg.get('avg_phase2_tokens', 0) * 100:.1f}% tokens, "
        f"{agg.get('avg_phase2_info', 0) * 100:.1f}% info"
    )
    print(
        f"  Phase 3 (Convergence):  {agg.get('avg_phase3_tokens', 0) * 100:.1f}% tokens, "
        f"{agg.get('avg_phase3_info', 0) * 100:.1f}% info"
    )
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
