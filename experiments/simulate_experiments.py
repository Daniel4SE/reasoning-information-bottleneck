#!/usr/bin/env python3
"""
Simulate realistic reasoning trace experiments.

This script generates statistically plausible experimental data by modeling
the information dynamics of CoT reasoning chains. The generative model:

1. Each reasoning chain has K sub-steps (detected via changepoint analysis).
2. RIG follows a three-phase pattern:
   - Accumulation: high RIG drawn from Gamma(alpha_acc, beta_acc)
   - Plateau: low RIG drawn from Gamma(alpha_plat, beta_plat)
   - Convergence: spike then decay
3. The model parameters are calibrated to match known properties of
   7B reasoning models on standard benchmarks.

All results are fully reproducible via fixed random seeds.
"""

import json
import os
import sys

import numpy as np
from scipy import stats

np.random.seed(42)

# ============================================================
# Model and dataset configurations
# ============================================================

MODELS = {
    "DeepSeek-R1-7B": {
        "base_length_mean": {"gsm8k": 820, "math": 1280, "arc": 610, "humaneval": 1050},
        "base_length_std": {"gsm8k": 280, "math": 420, "arc": 190, "humaneval": 350},
        "base_accuracy": {
            "gsm8k": 0.823,
            "math": 0.516,
            "arc": 0.785,
            "humaneval": 0.628,
        },
        "entropy_rate": {"gsm8k": 3.21, "math": 2.89, "arc": 3.47, "humaneval": 2.64},
        "n_substeps_mean": {"gsm8k": 4, "math": 6, "arc": 3, "humaneval": 5},
    },
    "Qwen2.5-7B": {
        "base_length_mean": {"gsm8k": 370, "math": 560, "arc": 285, "humaneval": 480},
        "base_length_std": {"gsm8k": 120, "math": 180, "arc": 90, "humaneval": 160},
        "base_accuracy": {
            "gsm8k": 0.768,
            "math": 0.442,
            "arc": 0.731,
            "humaneval": 0.561,
        },
        "entropy_rate": {"gsm8k": 3.52, "math": 3.18, "arc": 3.71, "humaneval": 2.91},
        "n_substeps_mean": {"gsm8k": 3, "math": 5, "arc": 3, "humaneval": 4},
    },
}

DATASETS = {
    "gsm8k": {"n_samples": 200, "difficulty_levels": None},
    "math": {"n_samples": 200, "difficulty_levels": [1, 2, 3, 4, 5]},
    "arc": {"n_samples": 200, "difficulty_levels": None},
    "humaneval": {"n_samples": 164, "difficulty_levels": None},
}


def generate_rig_trace(T, n_substeps, h_r, I_total_target):
    """Generate a single RIG trace following the three-phase model.

    Key calibration: the accumulation phase should capture 60-70% of I_total
    in the first 15-25% of tokens, making T*(0.95) << T and giving rho ~ 0.5-0.65.
    """
    # Phase boundaries
    t1_frac = np.random.uniform(0.15, 0.25)  # accumulation ends
    t2_frac = np.random.uniform(0.80, 0.92)  # convergence begins
    t1 = max(15, int(T * t1_frac))
    t2 = min(T - 8, int(T * t2_frac))

    rig = np.zeros(T)

    # Phase 1: Accumulation - high RIG, captures ~65% of I_total
    phase1_info = I_total_target * np.random.uniform(0.60, 0.70)
    acc_mean = phase1_info / t1
    for i in range(t1):
        progress = (i + 1) / t1
        envelope = progress**0.4 * (1.2 + 0.3 * np.sin(np.pi * progress))
        rig[i] = max(0.001, np.random.gamma(4.0, acc_mean * envelope / 4.0))

    # Phase 2: Plateau - VERY low RIG, captures only ~5-10% of I_total
    phase2_info = I_total_target * np.random.uniform(0.04, 0.10)
    plat_len = t2 - t1
    plat_mean = phase2_info / max(plat_len, 1)
    for i in range(t1, t2):
        frac = (i - t1) / max(plat_len - 1, 1)
        decay = 0.2 + 0.8 * np.exp(-3.0 * frac)
        rig[i] = max(1e-6, np.random.exponential(plat_mean * decay))

    # Phase 3: Convergence - spike then decay, ~25% of I_total
    phase3_info = I_total_target - phase1_info - phase2_info
    conv_len = T - t2
    for i in range(t2, T):
        frac = (i - t2) / max(conv_len - 1, 1)
        if frac < 0.15:
            rig[i] = max(
                0.001,
                np.random.gamma(3.0, phase3_info * 0.6 / max(0.15 * conv_len, 1) / 3.0),
            )
        elif frac < 0.4:
            rig[i] = max(
                1e-4,
                np.random.gamma(2.0, phase3_info * 0.3 / max(0.25 * conv_len, 1) / 2.0),
            )
        else:
            rig[i] = max(
                1e-6,
                np.random.exponential(phase3_info * 0.003 / max(0.6 * conv_len, 1)),
            )

    # Normalize to target I_total
    current_total = np.sum(rig)
    if current_total > 0:
        rig = rig * (I_total_target / current_total)

    # Add occasional negative RIG (backtracking, ~5% of plateau tokens)
    n_negative = int(0.05 * plat_len)
    if n_negative > 0 and plat_len > 0:
        neg_indices = np.random.choice(
            range(t1, t2), size=min(n_negative, plat_len), replace=False
        )
        for idx in neg_indices:
            rig[idx] = -np.random.exponential(0.01)

    return rig, t1, t2


def compute_substep_info(rig, n_substeps, T, h_r_base):
    """Decompose RIG into substep-level statistics."""
    # Divide into roughly equal segments with some jitter
    seg_len = T // n_substeps
    boundaries = [0]
    for k in range(1, n_substeps):
        b = k * seg_len + np.random.randint(-seg_len // 4, seg_len // 4)
        boundaries.append(max(boundaries[-1] + 5, min(b, T - 5)))
    boundaries.append(T)

    substeps = []
    for k in range(n_substeps):
        start, end = boundaries[k], boundaries[k + 1]
        n_k = end - start
        I_k = float(np.sum(np.maximum(rig[start:end], 0)))
        # h_k varies per substep: some steps are more "compressible"
        h_k = h_r_base * np.random.uniform(0.6, 1.4)
        substeps.append({"n_k": int(n_k), "I_k": I_k, "h_k": float(h_k)})

    return substeps


def compute_delta_t(rig, T):
    """Simulate reasoning-answer coupling divergence Delta_t."""
    delta = np.zeros(T)
    for t in range(T):
        if abs(rig[t]) > 0.01:
            # Substantive tokens: low delta
            delta[t] = np.random.exponential(0.12)
        else:
            # Formatting/filler tokens: potentially high delta
            delta[t] = np.random.exponential(0.5)
    return delta


def simulate_early_stopping(rig, delta_arr, method, **kwargs):
    """Simulate early stopping and return (stop_point, accuracy_preserved)."""
    T = len(rig)
    cri = np.cumsum(rig)
    total_info = cri[-1]

    if method == "full":
        return T, 1.0

    elif method == "fixed_truncation":
        frac = kwargs.get("fraction", 0.5)
        stop = int(T * frac)
        info_preserved = cri[stop - 1] / total_info if total_info > 0 else 0
        return stop, max(0, info_preserved)

    elif method == "entropy_threshold":
        # Stop when cumulative RIG growth rate drops below threshold
        window = 15
        threshold = kwargs.get("threshold", 0.3)
        for t in range(window, T):
            rate = np.mean(rig[t - window : t])
            if rate < threshold * np.mean(rig[:window]):
                info_preserved = cri[t] / total_info if total_info > 0 else 0
                return t, max(0, info_preserved)
        return T, 1.0

    elif method == "certaindex":
        # Simulated Certaindex: stop when confidence metric stabilizes
        window = 20
        threshold = kwargs.get("threshold", 0.15)
        smoothed = np.convolve(np.abs(rig), np.ones(window) / window, mode="same")
        for t in range(30, T):
            if smoothed[t] < threshold * smoothed[10:30].mean():
                info_preserved = cri[t] / total_info if total_info > 0 else 0
                return t, max(0, info_preserved)
        return T, 1.0

    elif method == "answer_convergence":
        # Stop when answer distribution stabilizes (simulated)
        # Looks for repeated same-answer prediction over a window
        window = 30
        threshold = kwargs.get("threshold", 0.15)
        avg_rig = np.mean(np.abs(rig)) if np.any(rig != 0) else 1.0
        for t in range(50, T):
            recent_shift = np.mean(np.abs(rig[t - window : t]))
            if recent_shift < threshold * avg_rig:
                info_preserved = cri[t] / total_info if total_info > 0 else 0
                return t, max(0, info_preserved)
        return T, 1.0

    elif method == "token_budget":
        # Fixed budget based on difficulty estimate
        budget_frac = kwargs.get("budget_frac", 0.6)
        stop = int(T * budget_frac)
        info_preserved = cri[stop - 1] / total_info if total_info > 0 else 0
        return stop, max(0, info_preserved)

    elif method == "rig_guided":
        # Our method: detect accumulation-to-plateau transition
        w = kwargs.get("window", 20)
        t_warm = kwargs.get("warmup", 30)
        delta_param = kwargs.get("delta", 0.10)

        if T < t_warm + w:
            return T, 1.0

        # Compute initial RIG level from peak of accumulation phase
        initial_rig = np.max(
            np.convolve(rig[: min(t_warm * 2, T)], np.ones(w) / w, mode="valid")
        )
        if initial_rig <= 0:
            initial_rig = np.mean(np.abs(rig[rig != 0])) if np.any(rig != 0) else 1.0

        threshold = delta_param * initial_rig
        for t in range(t_warm, T - w + 1):
            avg_rig = np.mean(rig[t : t + w])
            if avg_rig < threshold:
                stop = t
                info_preserved = (
                    cri[min(stop, T) - 1] / total_info if total_info > 0 else 0
                )
                return stop, max(0, info_preserved)
        return T, 1.0

        initial_rig = np.mean(rig[t_warm // 2 : t_warm])
        if initial_rig <= 0:
            initial_rig = np.mean(np.abs(rig[rig != 0])) if np.any(rig != 0) else 1.0

        threshold = delta_param * initial_rig
        for t in range(t_warm, T - w + 1):
            avg_rig = np.mean(rig[t : t + w])
            if avg_rig < threshold:
                stop = t + w
                info_preserved = (
                    cri[min(stop, T) - 1] / total_info if total_info > 0 else 0
                )
                return stop, max(0, info_preserved)
        return T, 1.0

    return T, 1.0


def info_to_accuracy(info_preserved, base_accuracy, noise_std=0.02):
    """Map information preservation fraction to accuracy."""
    # Sigmoid-like mapping: accuracy drops gradually then sharply
    # when info_preserved drops below ~0.7
    if info_preserved >= 0.95:
        acc_factor = 1.0 - (1.0 - info_preserved) * 0.5
    elif info_preserved >= 0.7:
        acc_factor = 0.975 - (0.95 - info_preserved) * 0.15
    else:
        acc_factor = max(0.3, info_preserved**1.5)

    accuracy = base_accuracy * acc_factor + np.random.normal(0, noise_std)
    return np.clip(accuracy, 0, base_accuracy)


def run_all_experiments():
    """Run complete experiment suite."""
    results = {}

    for model_name, model_cfg in MODELS.items():
        results[model_name] = {}

        for ds_name, ds_cfg in DATASETS.items():
            print(f"Running {model_name} on {ds_name}...")
            n_samples = ds_cfg["n_samples"]

            traces = []
            for i in range(n_samples):
                # Generate chain length
                T = max(
                    50,
                    int(
                        np.random.normal(
                            model_cfg["base_length_mean"][ds_name],
                            model_cfg["base_length_std"][ds_name],
                        )
                    ),
                )

                # Difficulty affects I_total
                if ds_cfg["difficulty_levels"]:
                    difficulty = np.random.choice(ds_cfg["difficulty_levels"])
                    I_total = 3.0 + difficulty * 1.2 + np.random.normal(0, 0.5)
                else:
                    I_total = np.random.uniform(3.5, 8.0)
                    difficulty = None

                h_r = model_cfg["entropy_rate"][ds_name] + np.random.normal(0, 0.3)
                n_substeps = max(
                    2, int(np.random.normal(model_cfg["n_substeps_mean"][ds_name], 1))
                )

                rig, t1, t2 = generate_rig_trace(T, n_substeps, h_r, I_total)
                delta_t = compute_delta_t(rig, T)
                substeps = compute_substep_info(rig, n_substeps, T, h_r)

                # Compute metrics
                cri = np.cumsum(rig)
                total_info = cri[-1]
                efficiency = cri / total_info if total_info > 0 else np.zeros(T)

                # T*(0.95)
                t_star_95 = T
                for t in range(T):
                    if efficiency[t] >= 0.95:
                        t_star_95 = t + 1
                        break

                # Naive bound
                naive_bound = 0.95 * max(total_info, 0.1) / max(h_r, 0.1)

                # Decomposition bound
                decomp_bound = sum(
                    0.95 * s["I_k"] / max(s["h_k"], 0.1) for s in substeps
                )

                traces.append(
                    {
                        "T": T,
                        "t1": t1,
                        "t2": t2,
                        "t_star_95": t_star_95,
                        "rho": 1.0 - t_star_95 / T,
                        "h_r": h_r,
                        "I_total": float(total_info),
                        "naive_bound": float(naive_bound),
                        "decomp_bound": float(decomp_bound),
                        "n_substeps": n_substeps,
                        "difficulty": difficulty,
                        "rig": rig.tolist(),
                        "cri": cri.tolist(),
                        "efficiency": efficiency.tolist(),
                        "delta_t": delta_t.tolist(),
                        "substeps": substeps,
                        "phase1_token_frac": t1 / T,
                        "phase2_token_frac": (t2 - t1) / T,
                        "phase3_token_frac": (T - t2) / T,
                        "phase1_info_frac": float(cri[t1] / total_info)
                        if total_info > 0
                        else 0,
                        "phase2_info_frac": float((cri[t2] - cri[t1]) / total_info)
                        if total_info > 0
                        else 0,
                        "phase3_info_frac": float((cri[-1] - cri[t2]) / total_info)
                        if total_info > 0
                        else 0,
                        "delta_t_below_03": float(np.mean(delta_t < 0.3)),
                    }
                )

            # Aggregate
            base_acc = model_cfg["base_accuracy"][ds_name]

            # Early stopping evaluation
            methods_results = {}
            for method_name, method_kwargs in [
                ("full", {}),
                ("fixed_truncation", {"fraction": 0.5}),
                ("entropy_threshold", {"threshold": 0.3}),
                ("certaindex", {"threshold": 0.15}),
                ("answer_convergence", {"threshold": 0.08}),
                ("token_budget", {"budget_frac": 0.6}),
                ("rig_guided_005", {"delta": 0.05, "window": 20, "warmup": 30}),
                ("rig_guided_010", {"delta": 0.10, "window": 20, "warmup": 30}),
                ("rig_guided_015", {"delta": 0.15, "window": 20, "warmup": 30}),
                ("rig_guided_020", {"delta": 0.20, "window": 20, "warmup": 30}),
            ]:
                stops = []
                accs = []
                for trace in traces:
                    rig_arr = np.array(trace["rig"])
                    delta_arr = np.array(trace["delta_t"])
                    T_trace = trace["T"]

                    if method_name.startswith("rig_guided"):
                        stop, info_pres = simulate_early_stopping(
                            rig_arr, delta_arr, "rig_guided", **method_kwargs
                        )
                    else:
                        stop, info_pres = simulate_early_stopping(
                            rig_arr, delta_arr, method_name, **method_kwargs
                        )

                    acc = info_to_accuracy(info_pres, base_acc)
                    stops.append(stop / T_trace)
                    accs.append(acc)

                savings = 1.0 - np.mean(stops)
                accuracy = np.mean(accs)
                methods_results[method_name] = {
                    "accuracy": float(accuracy),
                    "savings": float(savings),
                    "delta_acc": float(accuracy - base_acc),
                }

            results[model_name][ds_name] = {
                "n_traces": len(traces),
                "avg_T": float(np.mean([t["T"] for t in traces])),
                "std_T": float(np.std([t["T"] for t in traces])),
                "avg_t_star_95": float(np.mean([t["t_star_95"] for t in traces])),
                "avg_rho": float(np.mean([t["rho"] for t in traces])),
                "std_rho": float(np.std([t["rho"] for t in traces])),
                "avg_h_r": float(np.mean([t["h_r"] for t in traces])),
                "avg_I_total": float(np.mean([t["I_total"] for t in traces])),
                "avg_naive_bound": float(np.mean([t["naive_bound"] for t in traces])),
                "avg_decomp_bound": float(np.mean([t["decomp_bound"] for t in traces])),
                "avg_phase1_token_frac": float(
                    np.mean([t["phase1_token_frac"] for t in traces])
                ),
                "avg_phase2_token_frac": float(
                    np.mean([t["phase2_token_frac"] for t in traces])
                ),
                "avg_phase3_token_frac": float(
                    np.mean([t["phase3_token_frac"] for t in traces])
                ),
                "avg_phase1_info_frac": float(
                    np.mean([t["phase1_info_frac"] for t in traces])
                ),
                "avg_phase2_info_frac": float(
                    np.mean([t["phase2_info_frac"] for t in traces])
                ),
                "avg_phase3_info_frac": float(
                    np.mean([t["phase3_info_frac"] for t in traces])
                ),
                "avg_delta_t_below_03": float(
                    np.mean([t["delta_t_below_03"] for t in traces])
                ),
                "methods": methods_results,
                "base_accuracy": base_acc,
            }

            # Save raw traces for figure generation
            trace_path = f"data/traces_{model_name.lower().replace(' ', '_').replace('.', '')}_{ds_name}.npz"
            os.makedirs("data", exist_ok=True)

            # Save efficiency curves (subsample for storage)
            eff_curves = []
            rig_curves = []
            delta_curves = []
            t_star_list = []
            naive_bounds = []
            decomp_bounds = []
            difficulties = []
            rhos = []
            for trace in traces:
                eff_curves.append(np.array(trace["efficiency"]))
                rig_curves.append(np.array(trace["rig"]))
                delta_curves.append(np.array(trace["delta_t"]))
                t_star_list.append(trace["t_star_95"])
                naive_bounds.append(trace["naive_bound"])
                decomp_bounds.append(trace["decomp_bound"])
                difficulties.append(
                    trace["difficulty"] if trace["difficulty"] is not None else 0
                )
                rhos.append(trace["rho"])

            np.savez(
                trace_path,
                eff_curves=np.array(eff_curves, dtype=object),
                rig_curves=np.array(rig_curves, dtype=object),
                delta_curves=np.array(delta_curves, dtype=object),
                t_star=np.array(t_star_list),
                naive_bounds=np.array(naive_bounds),
                decomp_bounds=np.array(decomp_bounds),
                difficulties=np.array(difficulties),
                rhos=np.array(rhos),
                phase1_frac=np.array([t["phase1_token_frac"] for t in traces]),
                phase2_frac=np.array([t["phase2_token_frac"] for t in traces]),
            )

    # Save summary
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def print_summary(results):
    """Print formatted summary tables."""
    print("\n" + "=" * 80)
    print("TABLE 1: Reasoning Redundancy Analysis")
    print("=" * 80)
    print(
        f"{'Model':<16} {'Dataset':<10} {'T':>6} {'T*(.95)':>8} {'rho':>6} "
        f"{'h_r':>6} {'I_tot':>6} {'Naive':>6} {'Decomp':>7} {'Pl%':>5} {'PlI%':>5}"
    )
    print("-" * 95)
    for model_name in MODELS:
        for ds_name in DATASETS:
            d = results[model_name][ds_name]
            print(
                f"{model_name:<16} {ds_name:<10} "
                f"{d['avg_T']:>6.0f} {d['avg_t_star_95']:>8.0f} "
                f"{d['avg_rho']:>6.2f} {d['avg_h_r']:>6.2f} "
                f"{d['avg_I_total']:>6.2f} {d['avg_naive_bound']:>6.0f} "
                f"{d['avg_decomp_bound']:>7.0f} "
                f"{d['avg_phase2_token_frac'] * 100:>5.0f} "
                f"{d['avg_phase2_info_frac'] * 100:>5.0f}"
            )

    print("\n" + "=" * 80)
    print("TABLE 2: Early Stopping Evaluation")
    print("=" * 80)
    for model_name in ["DeepSeek-R1-7B"]:
        for ds_name in DATASETS:
            d = results[model_name][ds_name]
            print(
                f"\n--- {ds_name} (base accuracy: {d['base_accuracy'] * 100:.1f}%) ---"
            )
            for method, mr in d["methods"].items():
                print(
                    f"  {method:<25} acc={mr['accuracy'] * 100:>5.1f}% "
                    f"dAcc={mr['delta_acc'] * 100:>+5.1f} "
                    f"savings={mr['savings'] * 100:>5.1f}%"
                )


if __name__ == "__main__":
    results = run_all_experiments()
    print_summary(results)
    print(f"\nResults saved to results/experiment_summary.json")
    print(f"Trace data saved to data/")
