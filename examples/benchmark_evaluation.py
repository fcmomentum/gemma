#!/usr/bin/env python3
"""
Benchmark Evaluation Script for Memory-Trained Gemma Models

Evaluates trained checkpoints on standard LLM benchmarks:
- HellaSwag (commonsense reasoning)
- BoolQ (yes/no questions)
- PIQA (physical intuition)
- ARC-Easy/Challenge (science questions)
- WinoGrande (pronoun resolution)

Usage:
    python benchmark_evaluation.py --checkpoint /path/to/checkpoint.pkl
    python benchmark_evaluation.py --baseline  # Evaluate original Gemma3 1B
"""

import argparse
import os
import pickle
import json
from typing import Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

# Parse arguments first
parser = argparse.ArgumentParser(description='Evaluate Gemma model on benchmarks')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to trained checkpoint (.pkl file)')
parser.add_argument('--baseline', action='store_true',
                    help='Evaluate baseline Gemma3 1B (no fine-tuning)')
parser.add_argument('--output', type=str, default='eval_results.json',
                    help='Output file for results')
parser.add_argument('--max-examples', type=int, default=500,
                    help='Max examples per benchmark (for speed)')
parser.add_argument('--benchmark', type=str, default=None,
                    help='Run only specific benchmark (hellaswag, boolq, piqa, arc_easy, arc_challenge, winogrande)')
args = parser.parse_args()

# Import Gemma
from gemma import gm


@dataclass
class BenchmarkResult:
    """Results from a single benchmark."""
    name: str
    accuracy: float
    num_examples: int
    num_correct: int


class GemmaEvaluator:
    """Evaluator for Gemma models on standard benchmarks."""

    def __init__(self, params, model, tokenizer):
        self.params = params
        self.model = model
        self.tokenizer = tokenizer

        # Create jitted function with model captured in closure
        @jax.jit
        def _get_logits(params, input_ids):
            output = model.apply({'params': params}, input_ids)
            return output.logits

        self._get_logits_fn = _get_logits

    def get_logits(self, input_ids):
        """Get logits for input tokens."""
        return self._get_logits_fn(self.params, input_ids)

    def score_choices(self, prompt: str, choices: list[str]) -> int:
        """Score multiple choices and return index of best one."""
        scores = []

        for choice in choices:
            # Concatenate prompt + choice
            full_text = prompt + choice
            tokens = self.tokenizer.encode(full_text)
            prompt_tokens = self.tokenizer.encode(prompt)

            # Get logits
            input_ids = jnp.array([tokens[:-1]])  # All but last token
            logits = self.get_logits(input_ids)

            # Calculate log probability of choice tokens
            choice_start = len(prompt_tokens) - 1  # -1 because we shifted
            choice_logits = logits[0, choice_start:, :]
            choice_tokens = jnp.array(tokens[len(prompt_tokens):])

            # Log softmax and gather
            log_probs = jax.nn.log_softmax(choice_logits, axis=-1)
            token_log_probs = jnp.take_along_axis(
                log_probs, choice_tokens[:, None], axis=-1
            ).squeeze(-1)

            # Average log probability (length normalized)
            score = float(jnp.mean(token_log_probs))
            scores.append(score)

        return int(np.argmax(scores))

    def eval_hellaswag(self, max_examples: int = 500) -> BenchmarkResult:
        """Evaluate on HellaSwag (commonsense completion)."""
        try:
            from datasets import load_dataset
            dataset = load_dataset("Rowan/hellaswag", split="validation")
        except Exception as e:
            print(f"Could not load HellaSwag: {e}")
            return BenchmarkResult("hellaswag", 0.0, 0, 0)

        correct = 0
        total = min(len(dataset), max_examples)

        for i, example in enumerate(tqdm(dataset, total=total, desc="HellaSwag")):
            if i >= max_examples:
                break

            # Build prompt
            prompt = example["ctx"]
            choices = example["endings"]
            label = int(example["label"])

            pred = self.score_choices(prompt, choices)
            if pred == label:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        return BenchmarkResult("hellaswag", accuracy, total, correct)

    def eval_boolq(self, max_examples: int = 500) -> BenchmarkResult:
        """Evaluate on BoolQ (yes/no questions)."""
        try:
            from datasets import load_dataset
            dataset = load_dataset("boolq", split="validation")
        except Exception as e:
            print(f"Could not load BoolQ: {e}")
            return BenchmarkResult("boolq", 0.0, 0, 0)

        correct = 0
        total = min(len(dataset), max_examples)

        for i, example in enumerate(tqdm(dataset, total=total, desc="BoolQ")):
            if i >= max_examples:
                break

            # Build prompt
            passage = example["passage"]
            question = example["question"]
            label = 0 if example["answer"] else 1  # Yes=0, No=1

            prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
            choices = [" Yes", " No"]

            pred = self.score_choices(prompt, choices)
            if pred == label:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        return BenchmarkResult("boolq", accuracy, total, correct)

    def eval_piqa(self, max_examples: int = 500) -> BenchmarkResult:
        """Evaluate on PIQA (physical intuition)."""
        try:
            from datasets import load_dataset
            dataset = load_dataset("ybisk/piqa", split="validation")
        except Exception as e:
            print(f"Could not load PIQA: {e}")
            return BenchmarkResult("piqa", 0.0, 0, 0)

        correct = 0
        total = min(len(dataset), max_examples)

        for i, example in enumerate(tqdm(dataset, total=total, desc="PIQA")):
            if i >= max_examples:
                break

            prompt = f"Goal: {example['goal']}\nSolution:"
            choices = [f" {example['sol1']}", f" {example['sol2']}"]
            label = example["label"]

            pred = self.score_choices(prompt, choices)
            if pred == label:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        return BenchmarkResult("piqa", accuracy, total, correct)

    def eval_arc_easy(self, max_examples: int = 500) -> BenchmarkResult:
        """Evaluate on ARC-Easy (science questions)."""
        try:
            from datasets import load_dataset
            dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        except Exception as e:
            print(f"Could not load ARC-Easy: {e}")
            return BenchmarkResult("arc_easy", 0.0, 0, 0)

        correct = 0
        total = min(len(dataset), max_examples)

        for i, example in enumerate(tqdm(dataset, total=total, desc="ARC-Easy")):
            if i >= max_examples:
                break

            prompt = f"Question: {example['question']}\nAnswer:"
            choices = [f" {c}" for c in example["choices"]["text"]]
            label_key = example["answerKey"]
            # Convert A/B/C/D to 0/1/2/3
            label = ord(label_key) - ord('A') if label_key.isalpha() else int(label_key) - 1

            pred = self.score_choices(prompt, choices)
            if pred == label:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        return BenchmarkResult("arc_easy", accuracy, total, correct)

    def eval_arc_challenge(self, max_examples: int = 500, n_shot: int = 25) -> BenchmarkResult:
        """Evaluate on ARC-Challenge (harder science questions) with few-shot prompting."""
        try:
            from datasets import load_dataset
            # Load both train (for few-shot examples) and test (for evaluation)
            train_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
            test_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        except Exception as e:
            print(f"Could not load ARC-Challenge: {e}")
            return BenchmarkResult("arc_challenge", 0.0, 0, 0)

        # Build few-shot prompt from training examples
        few_shot_examples = []
        for i, ex in enumerate(train_dataset):
            if i >= n_shot:
                break
            q = ex['question']
            choices_text = ex['choices']['text']
            choices_labels = ex['choices']['label']
            answer_key = ex['answerKey']

            # Format choices as "A. choice1\nB. choice2\n..."
            choices_str = "\n".join([f"{lbl}. {txt}" for lbl, txt in zip(choices_labels, choices_text)])
            few_shot_examples.append(f"Question: {q}\n{choices_str}\nAnswer: {answer_key}")

        few_shot_prompt = "\n\n".join(few_shot_examples) + "\n\n"
        print(f"Using {n_shot}-shot prompt ({len(few_shot_prompt)} chars)")

        correct = 0
        total = min(len(test_dataset), max_examples)

        for i, example in enumerate(tqdm(test_dataset, total=total, desc=f"ARC-C ({n_shot}-shot)")):
            if i >= max_examples:
                break

            # Build prompt with few-shot context
            q = example['question']
            choices_text = example['choices']['text']
            choices_labels = example['choices']['label']
            choices_str = "\n".join([f"{lbl}. {txt}" for lbl, txt in zip(choices_labels, choices_text)])

            prompt = few_shot_prompt + f"Question: {q}\n{choices_str}\nAnswer:"

            # Score each choice (just the letter)
            choices = [f" {lbl}" for lbl in choices_labels]
            label_key = example["answerKey"]
            label = choices_labels.index(label_key) if label_key in choices_labels else 0

            pred = self.score_choices(prompt, choices)
            if pred == label:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        return BenchmarkResult("arc_challenge", accuracy, total, correct)

    def eval_winogrande(self, max_examples: int = 500) -> BenchmarkResult:
        """Evaluate on WinoGrande (pronoun resolution)."""
        try:
            from datasets import load_dataset
            dataset = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        except Exception as e:
            print(f"Could not load WinoGrande: {e}")
            return BenchmarkResult("winogrande", 0.0, 0, 0)

        correct = 0
        total = min(len(dataset), max_examples)

        for i, example in enumerate(tqdm(dataset, total=total, desc="WinoGrande")):
            if i >= max_examples:
                break

            sentence = example["sentence"]
            option1 = example["option1"]
            option2 = example["option2"]
            label = int(example["answer"]) - 1  # 1/2 -> 0/1

            # Replace _ with each option
            choice1 = sentence.replace("_", option1)
            choice2 = sentence.replace("_", option2)

            prompt = "Complete the sentence:\n"
            choices = [choice1, choice2]

            pred = self.score_choices(prompt, choices)
            if pred == label:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        return BenchmarkResult("winogrande", accuracy, total, correct)

    def run_all_benchmarks(self, max_examples: int = 500, only_benchmark: str = None) -> dict:
        """Run all benchmarks and return results."""
        results = {}

        benchmarks = [
            ("hellaswag", self.eval_hellaswag),
            ("boolq", self.eval_boolq),
            ("piqa", self.eval_piqa),
            ("arc_easy", self.eval_arc_easy),
            ("arc_challenge", self.eval_arc_challenge),
            ("winogrande", self.eval_winogrande),
        ]

        # Filter to single benchmark if specified
        if only_benchmark:
            benchmarks = [(n, fn) for n, fn in benchmarks if n == only_benchmark]
            if not benchmarks:
                print(f"Unknown benchmark: {only_benchmark}")
                return results

        for name, eval_fn in benchmarks:
            print(f"\n=== Evaluating {name} ===")
            result = eval_fn(max_examples)
            results[name] = {
                "accuracy": result.accuracy * 100,  # Convert to percentage
                "num_examples": result.num_examples,
                "num_correct": result.num_correct,
            }
            print(f"  Accuracy: {result.accuracy * 100:.1f}% ({result.num_correct}/{result.num_examples})")

        return results


def main():
    print("=" * 60)
    print("Gemma Benchmark Evaluation")
    print("=" * 60)

    # Load model
    print("\nLoading Gemma3 1B...")
    model = gm.nn.Gemma3_1B(tokens="input")
    tokenizer = gm.text.Gemma3Tokenizer()

    # Load params
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        with open(args.checkpoint, "rb") as f:
            params = pickle.load(f)
        model_name = f"checkpoint_{os.path.basename(args.checkpoint)}"
    else:
        print("Loading baseline Gemma3 1B weights...")
        params = gm.ckpts.load_params(
            path=gm.ckpts.CheckpointPath.GEMMA3_1B_IT,
        )
        model_name = "gemma3_1b_baseline"

    print(f"Model loaded: {model_name}")

    # Create evaluator
    evaluator = GemmaEvaluator(params, model, tokenizer)

    # Run benchmarks
    print(f"\nRunning benchmarks (max {args.max_examples} examples each)...")
    results = evaluator.run_all_benchmarks(args.max_examples, only_benchmark=args.benchmark)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print("-" * 40)

    for benchmark, metrics in results.items():
        print(f"  {benchmark:15} {metrics['accuracy']:5.1f}%")

    # Calculate average
    avg_accuracy = np.mean([m["accuracy"] for m in results.values()])
    print("-" * 40)
    print(f"  {'Average':15} {avg_accuracy:5.1f}%")

    # Save results
    output_data = {
        "model": model_name,
        "max_examples": args.max_examples,
        "results": results,
        "average_accuracy": avg_accuracy,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
