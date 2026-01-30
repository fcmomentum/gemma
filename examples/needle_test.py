# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Needle in a Haystack evaluation for Split-Brain Prophet.

This script tests the long-context memory capabilities of the Split-Brain model
by inserting a random fact ("needle") into a long context ("haystack") and
measuring retrieval accuracy at various context depths.

The "Student" stream in Split-Brain is designed to predict future states,
which should improve long-range information propagation.

Usage:

```sh
python examples/needle_test.py \
    --checkpoint=/path/to/checkpoint \
    --model_size=270m \
    --context_lengths=1024,2048,4096 \
    --depths=0.1,0.25,0.5,0.75,0.9 \
    --num_trials=10
```
"""

import argparse
import random
from typing import Any

from flax import linen as nn
from gemma import gm
from gemma.gm.nn import _split_brain
from gemma.gm.nn import _split_brain_transformer
import jax
import jax.numpy as jnp


# Needle facts to insert into the haystack
NEEDLE_FACTS = [
    "The secret password is: QUANTUM_BUTTERFLY_42",
    "The magic number for this document is: 73891",
    "Remember this code: ALPHA_CENTAURI_77",
    "The hidden key is: CRYSTAL_MOUNTAIN_99",
    "Important: The answer is EMERALD_PHOENIX_33",
]

# Retrieval questions
RETRIEVAL_QUESTIONS = {
    "QUANTUM_BUTTERFLY_42": "What is the secret password?",
    "73891": "What is the magic number mentioned in the document?",
    "ALPHA_CENTAURI_77": "What code should be remembered?",
    "CRYSTAL_MOUNTAIN_99": "What is the hidden key?",
    "EMERALD_PHOENIX_33": "What is the answer mentioned as important?",
}

# Haystack filler text (lorem ipsum style, but more coherent)
HAYSTACK_FILLER = """
The development of advanced language models has revolutionized natural language
processing. These models learn patterns from vast amounts of text data and can
generate coherent responses to various prompts. The training process involves
optimizing millions or billions of parameters to minimize prediction errors.

Modern architectures use attention mechanisms to capture long-range dependencies
in text. This allows the model to understand context from distant parts of the
input. The transformer architecture, introduced in 2017, became the foundation
for most state-of-the-art language models.

Fine-tuning pretrained models on specific tasks has become a common practice.
This approach leverages the general knowledge learned during pretraining while
adapting the model to specialized domains. The process is more efficient than
training from scratch and often yields better results.

Research continues to improve model efficiency and capabilities. New techniques
for training, inference, and deployment are constantly being developed. The
field of AI is evolving rapidly, with new breakthroughs announced regularly.
"""


def create_haystack_with_needle(
    needle: str,
    context_length: int,
    depth: float,
    tokenizer: Any,
) -> tuple[str, str]:
  """Create a haystack with a needle inserted at the specified depth.

  Args:
    needle: The fact to insert.
    context_length: Total context length in tokens.
    depth: Position to insert needle (0.0 = start, 1.0 = end).
    tokenizer: Tokenizer for length estimation.

  Returns:
    Tuple of (full_context, expected_answer).
  """
  # Estimate tokens per filler block
  filler_tokens = len(tokenizer.encode(HAYSTACK_FILLER))

  # Calculate how many filler blocks we need
  num_blocks = max(1, (context_length - 100) // filler_tokens)

  # Build haystack
  filler_blocks = [HAYSTACK_FILLER.strip()] * num_blocks

  # Insert needle at specified depth
  insert_position = int(len(filler_blocks) * depth)
  insert_position = max(0, min(insert_position, len(filler_blocks)))

  filler_blocks.insert(insert_position, f"\n\n{needle}\n\n")

  full_context = "\n".join(filler_blocks)

  # Extract expected answer from needle
  for answer, question in RETRIEVAL_QUESTIONS.items():
    if answer in needle:
      return full_context, answer

  return full_context, ""


def evaluate_retrieval(
    model: nn.Module,
    params: Any,
    tokenizer: Any,
    context: str,
    expected_answer: str,
    max_new_tokens: int = 50,
) -> tuple[bool, str]:
  """Evaluate if the model can retrieve the needle from the haystack.

  Args:
    model: The language model.
    params: Model parameters.
    tokenizer: Tokenizer.
    context: The haystack with needle.
    expected_answer: The expected answer to find.
    max_new_tokens: Maximum tokens to generate.

  Returns:
    Tuple of (success, generated_text).
  """
  # Find the question for this answer
  question = RETRIEVAL_QUESTIONS.get(expected_answer, "What was the special information mentioned?")

  # Create prompt
  prompt = f"""Read the following document carefully:

{context}

Based ONLY on the document above, answer this question:
{question}

Answer:"""

  # Tokenize
  input_tokens = tokenizer.encode(prompt)
  input_tokens = jnp.array([input_tokens])

  # Generate (simple greedy decoding)
  generated_tokens = []
  for _ in range(max_new_tokens):
    output = model.apply(
        {'params': params},
        input_tokens,
        deterministic=True,
    )
    next_token = jnp.argmax(output.logits[:, -1, :], axis=-1)
    generated_tokens.append(int(next_token[0]))

    # Check for EOS
    if next_token[0] == tokenizer.eos_id:
      break

    input_tokens = jnp.concatenate([input_tokens, next_token[:, None]], axis=1)

  generated_text = tokenizer.decode(generated_tokens)

  # Check if answer is in generated text
  success = expected_answer.lower() in generated_text.lower()

  return success, generated_text


def run_needle_test(
    model: nn.Module,
    params: Any,
    tokenizer: Any,
    context_lengths: list[int],
    depths: list[float],
    num_trials: int = 10,
) -> dict[tuple[int, float], float]:
  """Run the full needle in a haystack evaluation.

  Args:
    model: The language model.
    params: Model parameters.
    tokenizer: Tokenizer.
    context_lengths: List of context lengths to test.
    depths: List of depths to test (0.0 to 1.0).
    num_trials: Number of trials per (length, depth) combination.

  Returns:
    Dictionary mapping (context_length, depth) to accuracy.
  """
  results = {}

  for ctx_len in context_lengths:
    for depth in depths:
      successes = 0

      for trial in range(num_trials):
        # Pick a random needle
        needle = random.choice(NEEDLE_FACTS)

        # Create haystack
        context, expected_answer = create_haystack_with_needle(
            needle=needle,
            context_length=ctx_len,
            depth=depth,
            tokenizer=tokenizer,
        )

        # Evaluate
        success, _ = evaluate_retrieval(
            model=model,
            params=params,
            tokenizer=tokenizer,
            context=context,
            expected_answer=expected_answer,
        )

        if success:
          successes += 1

      accuracy = successes / num_trials
      results[(ctx_len, depth)] = accuracy
      print(f"Context: {ctx_len}, Depth: {depth:.2f} -> Accuracy: {accuracy:.1%}")

  return results


def print_results_table(results: dict[tuple[int, float], float], depths: list[float]) -> None:
  """Print results as a table."""
  context_lengths = sorted(set(k[0] for k in results.keys()))

  # Header
  print("\n" + "=" * 60)
  print("Needle in a Haystack Results")
  print("=" * 60)
  header = "Depth    | " + " | ".join(f"{d:.0%}".rjust(6) for d in depths)
  print(header)
  print("-" * len(header))

  # Rows
  for ctx_len in context_lengths:
    row_values = [results.get((ctx_len, d), 0.0) for d in depths]
    row = f"{ctx_len:>7} | " + " | ".join(f"{v:.0%}".rjust(6) for v in row_values)
    print(row)

  print("=" * 60)


def main():
  parser = argparse.ArgumentParser(description='Needle in a Haystack Evaluation')
  parser.add_argument('--model_size', type=str, default='270m',
                      choices=['270m', '1b'], help='Model size')
  parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to checkpoint (if None, uses random init)')
  parser.add_argument('--context_lengths', type=str, default='512,1024,2048',
                      help='Comma-separated context lengths')
  parser.add_argument('--depths', type=str, default='0.1,0.25,0.5,0.75,0.9',
                      help='Comma-separated depth values (0.0-1.0)')
  parser.add_argument('--num_trials', type=int, default=10,
                      help='Number of trials per (length, depth) pair')
  args = parser.parse_args()

  # Parse context lengths and depths
  context_lengths = [int(x) for x in args.context_lengths.split(',')]
  depths = [float(x) for x in args.depths.split(',')]

  print(f"Needle in a Haystack Evaluation")
  print(f"Model: {args.model_size}")
  print(f"Context lengths: {context_lengths}")
  print(f"Depths: {depths}")
  print(f"Trials per combination: {args.num_trials}")

  # Create model
  if args.model_size == '270m':
    base_config = gm.nn.Gemma3_270M.config
    split_layers = (13, 14, 15)
  else:
    base_config = gm.nn.Gemma3_1B.config
    split_layers = (19, 20, 21)

  sb_config = _split_brain.SplitBrainConfig(split_brain_layers=split_layers)
  model = _split_brain_transformer.SplitBrainTransformer(
      config=base_config,
      split_brain_config=sb_config,
  )

  # Initialize or load params
  rng = jax.random.PRNGKey(42)
  max_ctx = max(context_lengths)
  dummy_tokens = jnp.ones((1, max_ctx), dtype=jnp.int32)
  params = model.init(rng, dummy_tokens, deterministic=True)['params']

  if args.checkpoint:
    # TODO: Load checkpoint using Orbax
    print(f"Loading checkpoint from {args.checkpoint}...")
    # params = load_checkpoint(args.checkpoint)

  # Create tokenizer
  tokenizer = gm.text.Gemma3Tokenizer()

  # Run evaluation
  print("\nRunning evaluation...")
  results = run_needle_test(
      model=model,
      params=params,
      tokenizer=tokenizer,
      context_lengths=context_lengths,
      depths=depths,
      num_trials=args.num_trials,
  )

  # Print results table
  print_results_table(results, depths)


if __name__ == '__main__':
  main()
