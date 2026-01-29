#!/usr/bin/env python3
"""
Long-Context Memory Probe Test

Tests model's ability to recall information from beyond the sliding window.
This specifically measures if the memory loss helps retain information from
positions T-W and earlier, where W is the sliding window size (512 for Gemma 1B).

Test structure:
1. Insert a "needle" (key fact) at the start of context
2. Fill with "haystack" (distractor text)
3. Ask model to recall the needle at the end
4. Measure accuracy at different needle distances

Usage:
    python memory_probe.py --checkpoint /path/to/model.pkl
    python memory_probe.py --baseline
"""

import argparse
import os
import pickle
import random
from typing import Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser(description='Long-context memory probe test')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to trained checkpoint')
parser.add_argument('--baseline', action='store_true',
                    help='Evaluate baseline Gemma3 1B')
parser.add_argument('--num-trials', type=int, default=50,
                    help='Number of trials per distance')
parser.add_argument('--max-context', type=int, default=2048,
                    help='Maximum context length to test')
args = parser.parse_args()

# Import Gemma
from gemma import gm


@dataclass
class ProbeResult:
    """Result from a single probe distance."""
    distance: int  # How far back the needle was (in tokens)
    accuracy: float
    num_trials: int
    num_correct: int
    avg_log_prob: float  # Average log prob of correct answer


class MemoryProbe:
    """Probe for testing long-context memory."""

    # Needle facts organized by category for multiple-choice
    # Format: (name, answer, needle_sentence, question, distractors)
    NEEDLE_FACTS = [
        # Professions
        ("Alice", "engineer", "Alice works as an engineer.",
         "What is Alice's profession?", ["doctor", "teacher", "chef"]),
        ("Frank", "doctor", "Frank is a doctor.",
         "What is Frank's profession?", ["engineer", "teacher", "chef"]),
        ("Henry", "chef", "Henry works as a chef.",
         "What is Henry's profession?", ["engineer", "doctor", "teacher"]),
        ("Leo", "teacher", "Leo is a teacher.",
         "What is Leo's profession?", ["engineer", "doctor", "chef"]),

        # Cities
        ("Bob", "Paris", "Bob lives in Paris.",
         "Where does Bob live?", ["London", "Tokyo", "Berlin"]),
        ("Grace", "Tokyo", "Grace visited Tokyo.",
         "What city did Grace visit?", ["Paris", "London", "Berlin"]),
        ("Maya", "London", "Maya lives in London.",
         "Where does Maya live?", ["Paris", "Tokyo", "Berlin"]),

        # Instruments
        ("Carol", "piano", "Carol plays the piano.",
         "What instrument does Carol play?", ["violin", "guitar", "drums"]),
        ("Iris", "violin", "Iris plays the violin.",
         "What instrument does Iris play?", ["piano", "guitar", "drums"]),
        ("Noah", "guitar", "Noah plays the guitar.",
         "What instrument does Noah play?", ["piano", "violin", "drums"]),

        # Colors
        ("Emma", "blue", "Emma's favorite color is blue.",
         "What is Emma's favorite color?", ["green", "red", "yellow"]),
        ("Kate", "green", "Kate's favorite color is green.",
         "What is Kate's favorite color?", ["blue", "red", "yellow"]),

        # Numbers
        ("David", "42", "David's favorite number is 42.",
         "What is David's favorite number?", ["7", "99", "13"]),
        ("Jack", "99", "Jack's lucky number is 99.",
         "What is Jack's lucky number?", ["7", "42", "13"]),
        ("Olivia", "7", "Olivia's favorite number is 7.",
         "What is Olivia's favorite number?", ["42", "99", "13"]),
    ]

    # Filler text (book-like narrative)
    FILLER_SENTENCES = [
        "The sun rose slowly over the mountains.",
        "Birds sang in the morning light.",
        "The old house stood at the end of the lane.",
        "Leaves rustled in the gentle breeze.",
        "The river flowed quietly through the valley.",
        "Clouds drifted lazily across the sky.",
        "The forest was dark and mysterious.",
        "Stars twinkled in the night sky.",
        "The market was bustling with activity.",
        "Ships sailed across the calm sea.",
        "The castle towered over the village below.",
        "Musicians played in the town square.",
        "Children laughed and played in the park.",
        "The library held thousands of ancient books.",
        "Farmers worked diligently in the fields.",
        "The train departed from the station.",
        "Artists painted beautiful landscapes.",
        "The clock tower chimed at noon.",
        "Flowers bloomed in the garden.",
        "The mountain path was steep and winding.",
    ]

    def __init__(self, params, model, tokenizer):
        self.params = params
        self.model = model
        self.tokenizer = tokenizer

        # Detect available devices
        self.devices = jax.devices()
        self.n_devices = len(self.devices)
        print(f"Using {self.n_devices} device(s): {[d.platform for d in self.devices]}")

        # JIT compile forward pass (single device)
        @jax.jit
        def _get_logits(params, input_ids):
            output = model.apply({'params': params}, input_ids)
            return output.logits

        self._get_logits_fn = _get_logits

        # For multi-GPU: pmap version (data parallelism)
        if self.n_devices > 1:
            # Replicate params across devices
            self.replicated_params = jax.device_put_replicated(params, self.devices)

            @jax.pmap
            def _get_logits_parallel(params, input_ids):
                output = model.apply({'params': params}, input_ids)
                return output.logits

            self._get_logits_parallel_fn = _get_logits_parallel
            print(f"Multi-GPU mode enabled with pmap")
        else:
            self.replicated_params = None
            self._get_logits_parallel_fn = None

    def get_logits(self, input_ids):
        """Get model logits for input."""
        return self._get_logits_fn(self.params, input_ids)

    def get_logits_batch(self, batch_input_ids):
        """Get logits for a batch, using multi-GPU if available.

        Args:
            batch_input_ids: list of token ID lists (one per sample)
        Returns:
            list of logits arrays
        """
        if self.n_devices > 1 and len(batch_input_ids) >= self.n_devices:
            # Pad batch to be divisible by n_devices
            padded_size = ((len(batch_input_ids) + self.n_devices - 1) // self.n_devices) * self.n_devices
            while len(batch_input_ids) < padded_size:
                batch_input_ids.append(batch_input_ids[0])  # Pad with first example

            # Reshape for pmap: (n_devices, batch_per_device, seq_len)
            batch_per_device = len(batch_input_ids) // self.n_devices
            max_len = max(len(x) for x in batch_input_ids)

            # Pad sequences to same length
            padded_batch = []
            for ids in batch_input_ids:
                padded = ids + [0] * (max_len - len(ids))
                padded_batch.append(padded)

            # Reshape
            stacked = jnp.array(padded_batch).reshape(self.n_devices, batch_per_device, max_len)

            # Run parallel inference
            logits = self._get_logits_parallel_fn(self.replicated_params, stacked)

            # Flatten back
            return logits.reshape(-1, logits.shape[-2], logits.shape[-1])
        else:
            # Single GPU fallback
            results = []
            for ids in batch_input_ids:
                input_ids = jnp.array([ids])
                logits = self._get_logits_fn(self.params, input_ids)
                results.append(logits[0])
            return results

    def generate_filler(self, num_tokens: int) -> str:
        """Generate filler text of approximately num_tokens."""
        filler_parts = []
        current_tokens = 0

        while current_tokens < num_tokens:
            sentence = random.choice(self.FILLER_SENTENCES)
            filler_parts.append(sentence)
            # Rough estimate: ~1.3 tokens per word
            current_tokens += len(sentence.split()) * 1.3

        return " ".join(filler_parts)

    def create_probe(self, target_distance: int) -> tuple[str, list[str], int, str]:
        """Create a probe with needle at specified token distance.

        Returns: (full_context, choices, correct_idx, correct_answer)
        """
        # Pick a random needle (name, answer, needle_sentence, question, distractors)
        name, answer, needle_sentence, question, distractors = random.choice(self.NEEDLE_FACTS)

        # Calculate filler needed
        needle_tokens = len(self.tokenizer.encode(needle_sentence))
        filler_tokens = target_distance - needle_tokens

        if filler_tokens < 0:
            filler_tokens = 0

        # Generate filler
        filler = self.generate_filler(filler_tokens)

        # Construct context: needle -> filler -> question
        context = f"{needle_sentence} {filler}\n\nQuestion: {question}\nAnswer:"

        # Build choices (correct + 3 distractors), shuffle them
        all_choices = [answer] + list(distractors)
        random.shuffle(all_choices)
        correct_idx = all_choices.index(answer)

        return context, all_choices, correct_idx, answer

    def score_answer(self, context: str, answer: str) -> float:
        """Score the log probability of the answer given context."""
        # Tokenize context
        context_tokens = self.tokenizer.encode(context)
        answer_tokens = self.tokenizer.encode(" " + answer)  # Add space prefix

        # Combine
        full_tokens = context_tokens + answer_tokens
        input_ids = jnp.array([full_tokens[:-1]])  # Input tokens

        # Get logits
        logits = self.get_logits(input_ids)

        # Calculate log probs for answer tokens
        log_probs = jax.nn.log_softmax(logits[0], axis=-1)

        # Get log prob of each answer token
        answer_log_probs = []
        for i, token in enumerate(answer_tokens):
            pos = len(context_tokens) - 1 + i  # Position in sequence
            if pos < log_probs.shape[0]:
                answer_log_probs.append(float(log_probs[pos, token]))

        if not answer_log_probs:
            return float('-inf')

        return float(np.mean(answer_log_probs))

    def score_choices(self, context: str, choices: list[str]) -> int:
        """Score all choices and return index of highest scoring one."""
        scores = []
        for choice in choices:
            score = self.score_answer(context, choice)
            scores.append(score)
        return int(np.argmax(scores))

    def run_probe_at_distance(self, distance: int, num_trials: int = 50) -> ProbeResult:
        """Run probe test at a specific needle distance using multiple-choice."""
        correct = 0
        log_probs = []

        for _ in tqdm(range(num_trials), desc=f"Distance {distance}", leave=False):
            context, choices, correct_idx, correct_answer = self.create_probe(distance)

            # Multiple-choice scoring: pick the highest scoring choice
            predicted_idx = self.score_choices(context, choices)

            # Record accuracy
            if predicted_idx == correct_idx:
                correct += 1

            # Also record log prob of correct answer for analysis
            correct_score = self.score_answer(context, correct_answer)
            log_probs.append(correct_score)

        accuracy = correct / num_trials
        avg_log_prob = float(np.mean(log_probs))

        return ProbeResult(
            distance=distance,
            accuracy=accuracy,
            num_trials=num_trials,
            num_correct=correct,
            avg_log_prob=avg_log_prob,
        )

    def run_full_probe(self, distances: list[int], num_trials: int = 50) -> list[ProbeResult]:
        """Run probe at multiple distances."""
        results = []

        for dist in distances:
            print(f"\n=== Testing distance: {dist} tokens ===")
            result = self.run_probe_at_distance(dist, num_trials)
            results.append(result)
            print(f"  Accuracy: {result.accuracy * 100:.1f}%")
            print(f"  Avg Log Prob: {result.avg_log_prob:.3f}")

        return results


def main():
    print("=" * 60)
    print("Long-Context Memory Probe Test")
    print("=" * 60)

    # Load model
    model = gm.nn.Gemma3_1B(tokens="input")
    tokenizer = gm.text.Gemma3Tokenizer()

    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        with open(args.checkpoint, "rb") as f:
            params = pickle.load(f)
    else:
        print("\nLoading baseline Gemma3 1B PT...")
        params = gm.ckpts.load_params(path=gm.ckpts.CheckpointPath.GEMMA3_1B_PT)

    # Create probe
    probe = MemoryProbe(params, model, tokenizer)

    # Test at various distances
    # Key distances: within window (256, 512), at boundary (512-768), beyond (1024, 1536, 2048)
    distances = [
        256,   # Well within sliding window
        512,   # At sliding window boundary
        768,   # Just beyond window
        1024,  # ~2x window
        1536,  # ~3x window
        2048,  # ~4x window (if max_context allows)
    ]

    # Filter to max context
    distances = [d for d in distances if d <= args.max_context]

    print(f"\nTesting distances: {distances}")
    print(f"Trials per distance: {args.num_trials}")
    print(f"Sliding window size: 512 tokens")

    # Run probes
    results = probe.run_full_probe(distances, args.num_trials)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Distance':>10} {'Accuracy':>10} {'Log Prob':>10} {'Status':>15}")
    print("-" * 50)

    for r in results:
        status = "✓ In window" if r.distance <= 512 else "⚠ Beyond window"
        print(f"{r.distance:>10} {r.accuracy * 100:>9.1f}% {r.avg_log_prob:>10.3f} {status:>15}")

    # Key comparison
    in_window = [r for r in results if r.distance <= 512]
    beyond_window = [r for r in results if r.distance > 512]

    if in_window and beyond_window:
        in_window_acc = np.mean([r.accuracy for r in in_window])
        beyond_window_acc = np.mean([r.accuracy for r in beyond_window])

        print("\n" + "-" * 50)
        print(f"Avg accuracy IN window (≤512):     {in_window_acc * 100:.1f}%")
        print(f"Avg accuracy BEYOND window (>512): {beyond_window_acc * 100:.1f}%")
        print(f"Drop: {(in_window_acc - beyond_window_acc) * 100:.1f}%")
        print("\n💡 A good memory model should have smaller drop beyond the window!")


if __name__ == "__main__":
    main()
