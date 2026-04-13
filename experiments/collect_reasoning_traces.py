#!/usr/bin/env python3
"""
Collect reasoning traces with token-level logits from LLMs.

This script generates CoT reasoning chains and saves:
- The full reasoning text
- Token-level log-probabilities at each step
- The model's next-token distribution shift (for RIG computation)

Supports two backends:
1. MLX (Apple Silicon native, recommended for Mac)
2. Transformers (CPU/GPU, fallback)

Usage:
    python collect_reasoning_traces.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --dataset gsm8k \
        --output data/traces_deepseek_gsm8k.jsonl \
        --max-samples 200 \
        --max-tokens 2048 \
        --backend mlx
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def load_dataset(name: str, max_samples: int):
    """Load benchmark dataset."""
    if name == "gsm8k":
        try:
            from datasets import load_dataset

            ds = load_dataset("openai/gsm8k", "main", split="test")
            samples = []
            for i, ex in enumerate(ds):
                if i >= max_samples:
                    break
                # Extract numeric answer
                answer = ex["answer"].split("####")[-1].strip()
                samples.append(
                    {
                        "id": f"gsm8k_{i}",
                        "question": ex["question"],
                        "answer": answer,
                        "difficulty": "standard",
                    }
                )
            return samples
        except ImportError:
            print("datasets package not installed. Using built-in samples.")
            return _builtin_gsm8k_samples(max_samples)

    elif name == "math":
        try:
            from datasets import load_dataset

            ds = load_dataset("lighteval/MATH", "all", split="test")
            samples = []
            for i, ex in enumerate(ds):
                if i >= max_samples:
                    break
                samples.append(
                    {
                        "id": f"math_{i}",
                        "question": ex["problem"],
                        "answer": ex["solution"],
                        "difficulty": ex.get("level", "unknown"),
                    }
                )
            return samples
        except ImportError:
            return _builtin_math_samples(max_samples)

    elif name == "arc":
        try:
            from datasets import load_dataset

            ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
            samples = []
            for i, ex in enumerate(ds):
                if i >= max_samples:
                    break
                choices = ex["choices"]
                choices_text = " ".join(
                    f"({l}) {t}" for l, t in zip(choices["label"], choices["text"])
                )
                samples.append(
                    {
                        "id": f"arc_{i}",
                        "question": f"{ex['question']}\n{choices_text}",
                        "answer": ex["answerKey"],
                        "difficulty": "challenge",
                    }
                )
            return samples
        except ImportError:
            return _builtin_arc_samples(max_samples)

    else:
        raise ValueError(f"Unknown dataset: {name}")


def _builtin_gsm8k_samples(n):
    """Minimal built-in GSM8K examples for testing."""
    examples = [
        {
            "id": "gsm8k_0",
            "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
            "answer": "18",
            "difficulty": "easy",
        },
        {
            "id": "gsm8k_1",
            "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "answer": "3",
            "difficulty": "easy",
        },
        {
            "id": "gsm8k_2",
            "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            "answer": "70000",
            "difficulty": "medium",
        },
        {
            "id": "gsm8k_3",
            "question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
            "answer": "540",
            "difficulty": "easy",
        },
        {
            "id": "gsm8k_4",
            "question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
            "answer": "20",
            "difficulty": "medium",
        },
    ]
    return examples[:n]


def _builtin_math_samples(n):
    """Minimal built-in MATH examples."""
    examples = [
        {
            "id": "math_0",
            "question": "Find the value of $x$ such that $\\sqrt{x+7} = 9$.",
            "answer": "74",
            "difficulty": "Level 1",
        },
        {
            "id": "math_1",
            "question": "What is the sum of all values of $y$ for which the expression $\\frac{y+6}{y^2-5y+4}$ is undefined?",
            "answer": "5",
            "difficulty": "Level 3",
        },
    ]
    return examples[:n]


def _builtin_arc_samples(n):
    """Minimal built-in ARC examples."""
    examples = [
        {
            "id": "arc_0",
            "question": "Which of the following is an example of a physical change?\n(A) Rusting of iron (B) Burning of wood (C) Melting of ice (D) Digesting food",
            "answer": "C",
            "difficulty": "challenge",
        },
    ]
    return examples[:n]


def format_prompt(question: str, model_type: str = "reasoning") -> str:
    """Format a question as a prompt for reasoning."""
    if model_type == "reasoning":
        return (
            f"<|begin_of_thought|>\n"
            f"Solve the following problem step by step.\n\n"
            f"Problem: {question}\n\n"
            f"Think through this carefully:\n"
        )
    else:
        return (
            f"Solve the following problem step by step. "
            f"Show your reasoning, then give the final answer.\n\n"
            f"Problem: {question}\n\n"
            f"Solution:\n"
        )


def collect_with_mlx(model_name, samples, max_tokens, output_path):
    """Collect traces using MLX backend."""
    try:
        import mlx.core as mx
        from mlx_lm import load, generate
        from mlx_lm.utils import generate_step
    except ImportError:
        print("ERROR: mlx-lm not installed. Install with: pip install mlx-lm")
        sys.exit(1)

    print(f"Loading model {model_name} with MLX...")
    model, tokenizer = load(model_name)
    print("Model loaded.")

    results = []
    for idx, sample in enumerate(samples):
        print(f"[{idx + 1}/{len(samples)}] Processing {sample['id']}...")
        prompt = format_prompt(sample["question"])
        input_ids = mx.array(tokenizer.encode(prompt))

        # Collect token-level logits during generation
        token_logprobs = []
        kl_divergences = []
        generated_tokens = []
        prev_logits = None

        # Manual generation loop to capture logits
        prompt_tokens = tokenizer.encode(prompt)
        tokens = mx.array([prompt_tokens])

        for step in range(max_tokens):
            logits = model(tokens)
            # Get logits for the last position
            last_logits = logits[0, -1, :]

            # Compute softmax probabilities
            probs = mx.softmax(last_logits, axis=-1)
            log_probs = mx.log(probs + 1e-10)

            # Sample or greedy decode
            next_token = mx.argmax(last_logits).item()

            # Token log probability
            token_lp = log_probs[next_token].item()
            token_logprobs.append(token_lp)

            # Entropy of current distribution
            entropy = -mx.sum(probs * log_probs).item()

            # KL divergence from previous step (our RIG estimate)
            if prev_logits is not None:
                prev_probs = mx.softmax(prev_logits, axis=-1)
                # KL(current || previous) - top-k approximation
                top_k = 1000
                top_indices = mx.argsort(probs)[-top_k:]
                p = probs[top_indices]
                q = prev_probs[top_indices]
                kl = mx.sum(p * mx.log((p + 1e-10) / (q + 1e-10))).item()
                kl_divergences.append(kl)
            else:
                kl_divergences.append(0.0)

            prev_logits = last_logits

            generated_tokens.append(next_token)

            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                break

            # Append token and continue
            tokens = mx.concatenate([tokens, mx.array([[next_token]])], axis=1)

        # Decode generated text
        generated_text = tokenizer.decode(generated_tokens)

        result = {
            "id": sample["id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "difficulty": sample["difficulty"],
            "generated_text": generated_text,
            "num_tokens": len(generated_tokens),
            "token_logprobs": token_logprobs,
            "kl_divergences": kl_divergences,
        }
        results.append(result)

        # Save incrementally
        with open(output_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        print(f"  -> {len(generated_tokens)} tokens generated")

    return results


def collect_with_transformers(model_name, samples, max_tokens, output_path):
    """Collect traces using HuggingFace Transformers backend."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print(
            "ERROR: transformers not installed. Install with: pip install transformers torch"
        )
        sys.exit(1)

    print(f"Loading model {model_name} with Transformers...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")

    results = []
    for idx, sample in enumerate(samples):
        print(f"[{idx + 1}/{len(samples)}] Processing {sample['id']}...")
        prompt = format_prompt(sample["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        token_logprobs = []
        kl_divergences = []
        generated_tokens = []
        prev_probs = None

        input_ids = inputs["input_ids"]

        with torch.no_grad():
            for step in range(max_tokens):
                outputs = model(input_ids)
                logits = outputs.logits[0, -1, :]

                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log(probs + 1e-10)

                next_token = torch.argmax(logits).item()
                token_lp = log_probs[next_token].item()
                token_logprobs.append(token_lp)

                # KL divergence (RIG estimate)
                if prev_probs is not None:
                    top_k = 1000
                    top_indices = torch.topk(probs, top_k).indices
                    p = probs[top_indices]
                    q = prev_probs[top_indices]
                    kl = torch.sum(p * torch.log((p + 1e-10) / (q + 1e-10))).item()
                    kl_divergences.append(kl)
                else:
                    kl_divergences.append(0.0)

                prev_probs = probs.clone()
                generated_tokens.append(next_token)

                if next_token == tokenizer.eos_token_id:
                    break

                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_token]], device=model.device)],
                    dim=1,
                )

        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        result = {
            "id": sample["id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "difficulty": sample["difficulty"],
            "generated_text": generated_text,
            "num_tokens": len(generated_tokens),
            "token_logprobs": token_logprobs,
            "kl_divergences": kl_divergences,
        }
        results.append(result)

        with open(output_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        print(f"  -> {len(generated_tokens)} tokens generated")

    return results


def main():
    parser = argparse.ArgumentParser(description="Collect reasoning traces with logits")
    parser.add_argument(
        "--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k", "math", "arc"]
    )
    parser.add_argument("--output", type=str, default="data/traces.jsonl")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--backend", type=str, default="mlx", choices=["mlx", "transformers"]
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Clear output file
    open(args.output, "w").close()

    samples = load_dataset(args.dataset, args.max_samples)
    print(f"Loaded {len(samples)} samples from {args.dataset}")

    if args.backend == "mlx":
        collect_with_mlx(args.model, samples, args.max_tokens, args.output)
    else:
        collect_with_transformers(args.model, samples, args.max_tokens, args.output)

    print(f"Done. Results saved to {args.output}")


if __name__ == "__main__":
    main()
