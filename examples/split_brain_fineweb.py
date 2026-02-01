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

r"""Split-Brain Prophet training with FineWeb-Edu dataset.

This example trains a Split-Brain Prophet model on the FineWeb-Edu dataset,
which is ideal for improving reasoning and planning capabilities.

The Split-Brain architecture introduces:
- Teacher Stream: Standard causal attention
- Student Stream: Masked causal attention predicting Teacher's future states
- Combined Loss: CE (next-token) + λ * Prophet (Student predicts Teacher+1)

Train locally with Gemma3 270M:

```sh
python examples/split_brain_fineweb.py \
    --model_size=270m \
    --batch_size=8 \
    --max_length=1024 \
    --num_train_steps=10000 \
    --learning_rate=1e-4 \
    --prophet_weight=0.1 \
    --output_dir=/tmp/split_brain_270m
```

Train with Gemma3 1B:

```sh
python examples/split_brain_fineweb.py \
    --model_size=1b \
    --batch_size=4 \
    --max_length=1024 \
    --num_train_steps=10000 \
    --learning_rate=5e-5 \
    --prophet_weight=0.1 \
    --output_dir=/tmp/split_brain_1b
```
"""

import argparse
import functools
import os
import sys
import time
from typing import Any

# Ensure we import the local gemma package, not the installed one
# This allows access to the new _split_brain module that isn't in site-packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from flax import linen as nn
from flax.training import train_state, checkpoints
from gemma import gm
from gemma.gm.nn import _split_brain
from gemma.gm.nn import _split_brain_transformer
import jax
import jax.numpy as jnp

import optax
import wandb
from gemma.gm.ckpts import _checkpoint



# Layer indices where Split-Brain is applied (75% depth as per PRD)
# Gemma3 270M has 18 layers -> layers 13, 14, 15 (indices start at 0)
# Gemma3 1B has 26 layers -> layers 19, 20, 21
SPLIT_BRAIN_LAYERS_270M = (13, 14, 15)
SPLIT_BRAIN_LAYERS_1B = (19, 20, 21)


def get_model_and_config(
    model_size: str,
    split_brain_config: _split_brain.SplitBrainConfig,
    layer_indices_str: str | None = None,
) -> tuple[nn.Module, gm.nn.config.TransformerConfig]:
  """Get model and config for the specified size."""
  if model_size == '270m':
    base_config = gm.nn.Gemma3_270M.config
    default_layers = SPLIT_BRAIN_LAYERS_270M
  elif model_size == '1b':
    base_config = gm.nn.Gemma3_1B.config
    default_layers = SPLIT_BRAIN_LAYERS_1B
  else:
    raise ValueError(f'Unsupported model size: {model_size}')

  # Parse custom layers if provided
  if layer_indices_str is not None:
    if layer_indices_str.strip().lower() in ('none', 'empty', ''):
      split_layers = ()
    elif layer_indices_str.strip().lower() == 'default':
      split_layers = default_layers
    else:
      try:
        split_layers = tuple(int(x.strip()) for x in layer_indices_str.split(','))
      except ValueError:
        raise ValueError(f'Invalid layer indices: {layer_indices_str}')
  else:
    split_layers = default_layers

  # Create split-brain config with the appropriate layers
  sb_config = _split_brain.SplitBrainConfig(
      mask_ratio=split_brain_config.mask_ratio,
      prophet_weight=split_brain_config.prophet_weight,
      target_shift=split_brain_config.target_shift,
      stop_gradient=split_brain_config.stop_gradient,
      gate_init_bias=split_brain_config.gate_init_bias,
      split_brain_layers=split_layers,
      use_dino_loss=split_brain_config.use_dino_loss,
      student_temp=split_brain_config.student_temp,
      teacher_temp=split_brain_config.teacher_temp,
  )

  model = _split_brain_transformer.SplitBrainTransformer(
      config=base_config,
      split_brain_config=sb_config,
  )

  return model, base_config


def adapt_params(
    params: dict[str, Any],
    split_brain_layers: tuple[int, ...],
) -> dict[str, Any]:
  """Adapt standard Gemma params to Split-Brain architecture.

  For split-brain layers:
  1. Initialize teacher_attn with pretrained attn weights.
  2. Initialize student_attn with pretrained attn weights.
  3. Gated fusion and other new params remain uninitialized (will be merged later).
  """
  new_params = jax.tree_util.tree_map(lambda x: x, params)  # Copy

  # Navigate to transformer/layer_X
  if 'transformer' in new_params:
    transformer = new_params['transformer']
  else:
    # Handle flat or other structures if necessary, but _checkpoint.load_params
    # usually returns nested 'transformer' dict for our usage
    transformer = new_params

  for i in split_brain_layers:
    layer_name = f'layer_{i}'
    if layer_name not in transformer:
      continue

    layer = transformer[layer_name]

    # Check if this is a standard layer with 'attn'
    if 'attn' in layer:
      attn_params = layer.pop('attn')

      # Create split_attn structure
      split_attn = {
          'teacher': attn_params,
          'student': attn_params, # Initialize student with same weights
      }

      layer['split_attn'] = split_attn

      # Note: 'gated_fusion' params are not in pretrained checkpoint,
      # so they won't be in this dict. We rely on merging with initialized params.

  return new_params


def load_pretrained_params(
    model_size: str,
    split_brain_layers: tuple[int, ...],
    target_params_shape: Any,
) -> Any:
  """Download and adapt pretrained parameters."""

  # Map model size to Kaggle handle
  # Using instruction tuned variants as requested
  # Map model size to CheckpointPath
  # Using instruction tuned variants as requested
  if model_size == '270m':
    ckpt_path = gm.ckpts.CheckpointPath.GEMMA3_270M_IT
  elif model_size == '1b':
    ckpt_path = gm.ckpts.CheckpointPath.GEMMA3_1B_IT
  else:
    raise ValueError(f"Unknown model size for pretraining: {model_size}")

  print(f"Loading pretrained model from: {ckpt_path}...")

  # Load params using native gemma loader
  try:
    loaded_params = gm.ckpts.load_params(ckpt_path)
  except Exception as e:
    raise RuntimeError(
        f"Failed to load params from {ckpt_path}. "
        "Ensure you have access to the checkpoints (e.g. GCS setup). "
        f"Error: {e}"
    )

  print("Adapting parameters to Split-Brain architecture...")
  adapted_params = adapt_params(loaded_params, split_brain_layers)

  return adapted_params

  print("Adapting parameters to Split-Brain architecture...")
  adapted_params = adapt_params(loaded_params, split_brain_layers)

  return adapted_params


def merge_params(target_params, loaded_params):
    """Recursively merge loaded parameters into initialized target parameters.

    Values in loaded_params override target_params.
    Values present in target_params but missing in loaded_params are kept (random init).
    """
    if isinstance(target_params, dict) and isinstance(loaded_params, dict):
        new_dict = target_params.copy()
        for k, v in loaded_params.items():
            if k in new_dict:
                new_dict[k] = merge_params(new_dict[k], v)
            else:
                # Extra params in loaded (e.g. maybe unused internals), we accept them
                # or we could ignore them. For now, let's keep valid keys.
                 pass
        return new_dict
    else:
        return loaded_params



def create_tokenizer(model_size: str) -> gm.text.Tokenizer:
  """Create tokenizer for Gemma3 models."""
  return gm.text.Gemma3Tokenizer()


def prepare_fineweb_dataset(
    tokenizer: gm.text.Tokenizer,
    max_length: int,
    split: str = 'train',
    num_samples: int | None = None,
    skip: int = 0,
):
  """Load and prepare FineWeb-Edu dataset.

  Args:
    tokenizer: Gemma tokenizer.
    max_length: Maximum sequence length.
    split: Dataset split ('train' or 'test').
    num_samples: Optional limit on number of samples.
    skip: Number of samples to skip from the start.

  Returns:
    Iterator of tokenized batches.
  """
  # Load FineWeb-Edu from HuggingFace
  # Using streaming to avoid downloading the full dataset
  dataset = load_dataset(
      'HuggingFaceFW/fineweb-edu',
      name='sample-10BT',  # 10B token sample for faster iteration
      split=split,
      streaming=True,
  )

  if skip > 0:
    dataset = dataset.skip(skip)

  if num_samples:
    dataset = dataset.take(num_samples)

  def tokenize_fn(example):
    text = example['text']
    tokens = tokenizer.encode(text)
    # Truncate or pad to max_length
    if len(tokens) > max_length:
      tokens = tokens[:max_length]
    else:
      tokens = tokens + [0] * (max_length - len(tokens))
    return {'tokens': tokens}

  return dataset.map(tokenize_fn)


def collate_batch(examples: list[dict], max_length: int) -> dict[str, jnp.ndarray]:
  """Collate examples into a batch."""
  tokens = jnp.array([ex['tokens'][:max_length] for ex in examples])
  # Input is tokens[:-1], target is tokens[1:]
  inputs = tokens[:, :-1]
  targets = tokens[:, 1:]
  # Loss mask: 1 for real tokens, 0 for padding
  loss_mask = (targets != 0).astype(jnp.float32)
  return {
      'input': inputs,
      'target': targets,
      'loss_mask': loss_mask,
  }


def compute_loss(
    params: Any,
    batch: dict[str, jnp.ndarray],
    rng: jax.random.PRNGKey,
    model: nn.Module,
    prophet_weight: float,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
  """Compute combined CE + Prophet loss.

  Args:
    params: Model parameters.
    model: SplitBrainTransformer model.
    batch: Tokenized batch with 'input', 'target', 'loss_mask'.
    prophet_weight: Weight λ for auxiliary loss.
    rng: Random key for dropout/masking.

  Returns:
    Tuple of (total_loss, metrics_dict).
  """
  # Forward pass
  output = model.apply(
      {'params': params},
      batch['input'],
      deterministic=False,
      rngs={'dropout': rng},
  )

  # Cross-entropy loss (next-token prediction)
  logits = output.logits
  targets = batch['target']
  loss_mask = batch['loss_mask']

  # Compute per-token CE loss
  ce_loss = optax.softmax_cross_entropy_with_integer_labels(
      logits, targets
  )
  ce_loss = jnp.sum(ce_loss * loss_mask) / jnp.sum(loss_mask)

  # Auxiliary Prophet loss (came back as [B], need scalar)
  aux_loss = jnp.mean(output.aux_loss)

  # Combined loss
  total_loss = ce_loss + prophet_weight * aux_loss

  metrics = {
      'ce_loss': ce_loss,
      'aux_loss': aux_loss,
      'total_loss': total_loss,
  }

  return total_loss, metrics


def train_step(
    state: train_state.TrainState,
    batch: dict[str, jnp.ndarray],
    rng: jax.random.PRNGKey,
    model: nn.Module,
    prophet_weight: float,
) -> tuple[train_state.TrainState, dict[str, jnp.ndarray]]:
  """Single training step."""
  loss_fn = functools.partial(
      compute_loss,
      model=model,
      batch=batch,
      prophet_weight=prophet_weight,
      rng=rng,
  )

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, metrics), grads = grad_fn(state.params)

  state = state.apply_gradients(grads=grads)

  return state, metrics


def eval_step(
    params: Any,
    batch: dict[str, jnp.ndarray],
    model: nn.Module,
) -> dict[str, jnp.ndarray]:
  """Single evaluation step (deterministic, no gradient)."""
  # Forward pass with deterministic=True (no random masking)
  output = model.apply(
      {'params': params},
      batch['input'],
      deterministic=True,
  )

  # Cross-entropy loss
  logits = output.logits
  targets = batch['target']
  loss_mask = batch['loss_mask']

  ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
  ce_loss = jnp.sum(ce_loss * loss_mask) / jnp.sum(loss_mask)

  # Perplexity
  ppl = jnp.exp(ce_loss)

  return {
      'val_ce_loss': ce_loss,
      'val_ppl': ppl,
      'val_aux_loss': jnp.mean(output.aux_loss),
  }


def run_validation(
    state: train_state.TrainState,
    val_dataset,
    model: nn.Module,
    batch_size: int,
    max_length: int,
    num_val_batches: int = 50,
) -> dict[str, float]:
  """Run validation on a subset of the validation set."""
  jit_eval_step = jax.jit(functools.partial(eval_step, model=model))

  val_metrics = {'val_ce_loss': [], 'val_ppl': [], 'val_aux_loss': []}
  batch_buffer = []
  batch_count = 0

  for example in val_dataset:
    batch_buffer.append(example)

    if len(batch_buffer) >= batch_size:
      batch = collate_batch(batch_buffer[:batch_size], max_length - 1)
      batch_buffer = batch_buffer[batch_size:]

      metrics = jit_eval_step(state.params, batch)
      for k, v in metrics.items():
        val_metrics[k].append(float(v))

      batch_count += 1
      if batch_count >= num_val_batches:
        break

  # Average metrics
  return {k: sum(v) / len(v) if v else 0.0 for k, v in val_metrics.items()}


def main():
  parser = argparse.ArgumentParser(description='Train Split-Brain Prophet')
  parser.add_argument('--model_size', type=str, default='270m',
                      choices=['270m', '1b'], help='Model size')
  parser.add_argument('--batch_size', type=int, default=8)
  parser.add_argument('--max_length', type=int, default=1024)
  parser.add_argument('--num_train_steps', type=int, default=10000)
  parser.add_argument('--learning_rate', type=float, default=1e-4)
  parser.add_argument('--prophet_weight', type=float, default=0.1,
                      help='Weight λ for auxiliary Prophet loss')
  parser.add_argument('--mask_ratio', type=float, default=0.5,
                      help='Random masking ratio for Student stream')
  parser.add_argument('--target_shift', type=int, default=1,
                      help='Look-ahead shift for Prophet loss (default 1)')
  parser.add_argument('--use_dino_loss', action='store_true',
                      help='Use DINO-style Cross-Entropy loss instead of MSE')
  parser.add_argument('--student_temp', type=float, default=0.1,
                      help='Student temperature for DINO loss')
  parser.add_argument('--teacher_temp', type=float, default=0.04,
                      help='Teacher temperature for DINO loss')
  parser.add_argument('--split_brain_layers', type=str, default=None,
                      help='Comma-separated layer indices (e.g., "13,14,15"). Defaults to 75% depth.')
  parser.add_argument('--use_pretrained', action='store_true',
                      help='Initialize with pretrained Gemma 3 weights')
  parser.add_argument('--output_dir', type=str, default='/tmp/split_brain')
  parser.add_argument('--restore_dir', type=str, default=None,
                      help='Directory to restore checkpoint from')
  parser.add_argument('--log_every', type=int, default=100)
  parser.add_argument('--save_every', type=int, default=1000)
  parser.add_argument('--eval_every', type=int, default=500,
                      help='Run validation every N steps')
  parser.add_argument('--num_val_batches', type=int, default=50,
                      help='Number of batches for validation')
  parser.add_argument('--wandb_project', type=str, default='split-brain-prophet',
                      help='W&B project name (set to empty to disable)')
  parser.add_argument('--wandb_run_name', type=str, default=None,
                      help='W&B run name (auto-generated if not set)')
  parser.add_argument('--jax_cache_dir', type=str, default='/tmp/jax_cache',
                      help='Directory to store JAX compilation cache')
  args = parser.parse_args()

  # Enable JAX compilation cache
  jax.config.update('jax_compilation_cache_dir', args.jax_cache_dir)
  print(f'JAX compilation cache enabled at: {args.jax_cache_dir}')

  # Initialize W&B
  if args.wandb_project:
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f'split-brain-{args.model_size}',
        config={
            'model_size': args.model_size,
            'batch_size': args.batch_size,
            'max_length': args.max_length,
            'num_train_steps': args.num_train_steps,
            'learning_rate': args.learning_rate,
            'prophet_weight': args.prophet_weight,
            'mask_ratio': args.mask_ratio,
            'target_shift': args.target_shift,
            'use_dino_loss': args.use_dino_loss,
            'student_temp': args.student_temp,
            'teacher_temp': args.teacher_temp,
        },
    )

  print(f'Training Split-Brain Prophet with {args.model_size} model')
  print(f'Prophet weight λ = {args.prophet_weight}')
  print(f'Mask ratio = {args.mask_ratio}')
  print(f'Target shift = {args.target_shift}')
  print(f'Use DINO loss = {args.use_dino_loss}')
  if args.use_dino_loss:
    print(f'Temperature (S/T) = {args.student_temp}/{args.teacher_temp}')

  # Initialize random keys
  rng = jax.random.PRNGKey(42)
  rng, init_rng, data_rng = jax.random.split(rng, 3)

  # Create Split-Brain config
  split_brain_config = _split_brain.SplitBrainConfig(
      mask_ratio=args.mask_ratio,
      prophet_weight=args.prophet_weight,
      target_shift=args.target_shift,
      use_dino_loss=args.use_dino_loss,
      student_temp=args.student_temp,
      teacher_temp=args.teacher_temp,
  )

  # Create model
  model, base_config = get_model_and_config(
      args.model_size,
      split_brain_config,
      args.split_brain_layers
  )
  print(f'Model has {base_config.num_layers} layers')
  print(f'Split-Brain applied at layers: {model.split_brain_config.split_brain_layers}')

  # Create tokenizer and datasets (train + validation)
  tokenizer = create_tokenizer(args.model_size)

  # Train dataset (starts from 0)
  train_dataset = prepare_fineweb_dataset(
      tokenizer=tokenizer,
      max_length=args.max_length,
      split='train',
      skip=0,
  )

  # Validation dataset (starts after training + buffer)
  # Ensure we validate on "future" data not seen during training
  total_train_samples = args.num_train_steps * args.batch_size
  val_buffer = 10000
  val_start_offset = total_train_samples + val_buffer
  num_val_samples = args.num_val_batches * args.batch_size * 2

  val_dataset = prepare_fineweb_dataset(
      tokenizer=tokenizer,
      max_length=args.max_length,
      split='train',
      skip=val_start_offset,
      num_samples=num_val_samples,
  )

  # Initialize model (random init)
  dummy_tokens = jnp.ones((1, args.max_length - 1), dtype=jnp.int32)
  params = model.init(init_rng, dummy_tokens, deterministic=True)['params']

  # Load pretrained weights if requested
  if args.use_pretrained:
    adapted_params = load_pretrained_params(
        args.model_size,
        model.split_brain_config.split_brain_layers,
        params
    )
    # Merge: keep random init for new params (e.g. gates), overwrite shared ones
    # We use a custom merge or rely on the fact that dict structures match
    # except for missing keys in adapted_params.

    # Simple recursive merge helper
    def recursive_merge(target, source):
      if isinstance(target, dict):
        # If source is also dict, recurse
        if isinstance(source, dict):
          for k, v in source.items():
             if k in target:
               target[k] = recursive_merge(target[k], v)
        return target
      else:
        # Leaf: if source provided, use it
        return source

    print("Merging pretrained weights...")
    params = recursive_merge(params, adapted_params)

  # Count parameters
  param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
  print(f'Total parameters: {param_count:,}')

  # Create optimizer and training state
  optimizer = optax.adamw(
      learning_rate=optax.warmup_cosine_decay_schedule(
          init_value=0.0,
          peak_value=args.learning_rate,
          warmup_steps=500,
          decay_steps=args.num_train_steps,
          end_value=args.learning_rate * 0.1,
      ),
      weight_decay=0.01,
  )

  # Ensure all parameters are on the same device (e.g. TPU:0) to avoid mismatch
  # This handles cases where loaded params are sharded but new params are not.
  print("Ensuring parameter device consistency...")
  params = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices()[0]), params)

  state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=optimizer,
  )

  # JIT compile train step
  jit_train_step = jax.jit(
      functools.partial(train_step, model=model, prophet_weight=args.prophet_weight)
  )

  # Restore checkpoint if requested
  if args.restore_dir:
    print(f'Restoring checkpoint from {args.restore_dir}...')
    state = checkpoints.restore_checkpoint(ckpt_dir=args.restore_dir, target=state)
    print(f'Resumed from step {int(state.step)}')

  # Training loop
  print('Starting training...')
  batch_buffer = []
  step = int(state.step)
  start_time = time.time()

  # If we resumed, we need to fast-forward the dataset to where we left off
  # The original train_dataset started at skip=0.
  # We should skip (step * batch_size) samples.
  if step > 0:
      print(f"Fast-forwarding training data by {step * args.batch_size} samples...")
      train_dataset = prepare_fineweb_dataset(
          tokenizer=tokenizer,
          max_length=args.max_length,
          split='train',
          skip=step * args.batch_size,
      )

  for example in train_dataset:
    batch_buffer.append(example)

    if len(batch_buffer) >= args.batch_size:
      # Create batch
      batch = collate_batch(batch_buffer[:args.batch_size], args.max_length - 1)
      batch_buffer = batch_buffer[args.batch_size:]

      # Train step
      rng, step_rng = jax.random.split(rng)
      state, metrics = jit_train_step(state, batch, step_rng)

      step += 1

      if step % args.log_every == 0:
        elapsed = time.time() - start_time
        samples_per_sec = (step * args.batch_size) / elapsed
        print(
            f'Step {step}/{args.num_train_steps} | '
            f'CE: {float(metrics["ce_loss"]):.4f} | '
            f'Aux: {float(metrics["aux_loss"]):.4f} | '
            f'Total: {float(metrics["total_loss"]):.4f} | '
            f'{samples_per_sec:.1f} samples/sec'
        )
        # Log to W&B
        if args.wandb_project:
          wandb.log({
              'train/ce_loss': float(metrics['ce_loss']),
              'train/aux_loss': float(metrics['aux_loss']),
              'train/total_loss': float(metrics['total_loss']),
              'train/samples_per_sec': samples_per_sec,
          }, step=step)

      if step % args.save_every == 0:
        # Save checkpoint
        print(f'Saving checkpoint at step {step}...')
        checkpoints.save_checkpoint(
            ckpt_dir=args.output_dir,
            target=state,
            step=step,
            keep=3,
            overwrite=True
        )

      # Run validation
      if step % args.eval_every == 0:
        print(f'\n=== Validation at step {step} ===')
        val_metrics = run_validation(
            state=state,
            val_dataset=val_dataset,
            model=model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_val_batches=args.num_val_batches,
        )
        print(
            f'Val CE: {float(val_metrics["val_ce_loss"]):.4f} | '
            f'Val PPL: {float(val_metrics["val_ppl"]):.2f} | '
            f'Val Aux: {float(val_metrics["val_aux_loss"]):.4f}'
        )
        print('=' * 40 + '\n')
        # Log validation to W&B
        if args.wandb_project:
          wandb.log({
              'val/ce_loss': val_metrics['val_ce_loss'],
              'val/ppl': val_metrics['val_ppl'],
              'val/aux_loss': val_metrics['val_aux_loss'],
          }, step=step)

      if step >= args.num_train_steps:
        break

  print(f'Training complete! Final step: {step}')
  print(f'Final CE loss: {float(metrics["ce_loss"]):.4f}')
  print(f'Final Aux loss: {float(metrics["aux_loss"]):.4f}')

  # Finish W&B run
  # Finish W&B run
  # if args.wandb_project:
  #   wandb.finish()


if __name__ == '__main__':
  main()
