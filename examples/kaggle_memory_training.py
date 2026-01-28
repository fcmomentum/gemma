# Split-Brain Memory Training on Kaggle
# =====================================
# This script trains Gemma with Split-Brain memory heads using PG-19 dataset.
#
# Setup:
# 1. Add Kaggle dataset: "the-pg19-language-modeling-benchmark-dataset"
# 2. Enable TPU accelerator
# 3. Set WANDB_API_KEY in Kaggle secrets
# 4. Run: uv run kaggle_memory_training.py

import os
import glob
import functools
from typing import Iterator

# Don't force JAX_PLATFORMS - let it auto-detect TPU/GPU/CPU
# The system needs libtpu.so for TPU (only available in system Python, not uv venv)

import jax
import jax.numpy as jnp
import optax
import numpy as np

# Check devices
print(f"JAX devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")
print(f"Default backend: {jax.default_backend()}")

if jax.default_backend() == 'cpu':
    print("WARNING: Running on CPU! Use system Python (not uv) for TPU access.")

# Initialize WandB (uses WANDB_API_KEY from environment)
import wandb
wandb.init(
    project="gemma-memory",
    name="pg19-splitbrain",
    config={
        "model": "Gemma3_1B",
        "max_length": 1024,
        "batch_size": 8,
        "memory_loss_weight": 0.1,
        "window_size": 512,
    }
)

# === Load PG-19 Dataset ===
PG19_PATH = "/kaggle/input/the-pg19-language-modeling-benchmark-dataset"

def load_pg19_books(split: str = "train", max_books: int = None) -> list[str]:
    """Load book texts from PG-19 dataset."""
    split_path = os.path.join(PG19_PATH, split)
    print(f"Looking for books in: {split_path}")

    # List immediate files and subdirs
    if os.path.exists(split_path):
        print(f"Contents: {os.listdir(split_path)[:10]}...")

    # Find all txt files recursively (handles nested structure)
    texts = []
    for root, dirs, files in os.walk(split_path):
        for f in files:
            if f.endswith('.txt'):
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as fp:
                        content = fp.read()
                        if len(content) > 1000:  # Skip very short files
                            texts.append(content)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

                if max_books and len(texts) >= max_books:
                    break
        if max_books and len(texts) >= max_books:
            break

    print(f"Loaded {len(texts)} books from {split}")
    return texts

# Load a subset for testing
train_texts = load_pg19_books("train", max_books=100)  # Start small
print(f"Total characters: {sum(len(t) for t in train_texts):,}")

# === Tokenizer and Data Processing ===
from gemma import gm

tokenizer = gm.text.Gemma3Tokenizer()  # Use Gemma3 tokenizer

# Configuration for Gemma3 1B
MAX_LENGTH = 1024  # Sequence length (must be > window_size for memory loss)
WINDOW_SIZE = 512   # Gemma3 1B sliding window
BATCH_SIZE = 8      # Larger batch for smaller model

def create_training_examples(texts: list[str], max_length: int) -> Iterator[dict]:
    """Convert book texts to training examples."""
    for text in texts:
        # Tokenize
        tokens = tokenizer.encode(text)

        # Create overlapping chunks
        stride = max_length // 2  # 50% overlap
        for i in range(0, len(tokens) - max_length, stride):
            chunk = tokens[i : i + max_length]
            yield {
                "input": np.array(chunk[:-1], dtype=np.int32),
                "target": np.array(chunk[1:], dtype=np.int32),
                "loss_mask": np.ones(len(chunk) - 1, dtype=np.float32),
            }

# Create examples
examples = list(create_training_examples(train_texts, MAX_LENGTH))
print(f"Created {len(examples)} training examples")

# === Load Split-Brain Memory Model ===
model = gm.nn.Gemma3_1B(tokens="input")

# Load pretrained weights
params = gm.ckpts.load_params(
    path=gm.ckpts.CheckpointPath.GEMMA3_1B_IT,
)
print("Gemma3 1B loaded successfully!")

# === Define Loss Functions ===
def cross_entropy_loss(logits, targets, mask):
    """Standard next-token prediction loss."""
    vocab_size = logits.shape[-1]
    one_hot = jax.nn.one_hot(targets, vocab_size)
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(one_hot * log_probs, axis=-1)
    return jnp.sum(loss * mask) / jnp.sum(mask)

def memory_reconstruction_loss(hidden_states, window_size: int = 512):
    """L1 loss for memory head training.

    Trains model to predict state of token T-W from current state T.
    """
    if hidden_states is None:
        return jnp.array(0.0)

    seq_len = hidden_states.shape[1]
    if seq_len <= window_size:
        return jnp.array(0.0)

    # Current state (time T)
    current = hidden_states[:, window_size:, :]
    # Target state (time T-W) - stop gradient!
    target = jax.lax.stop_gradient(hidden_states[:, :-window_size, :])

    # L1 loss
    return jnp.mean(jnp.abs(current - target))

# === Training Setup ===
optimizer = optax.adafactor(learning_rate=1e-4)
opt_state = optimizer.init(params)

MEMORY_LOSS_WEIGHT = 0.1
EFFECTIVE_WINDOW = min(WINDOW_SIZE, MAX_LENGTH // 2)  # Adjust for our sequence length

@functools.partial(jax.jit, donate_argnums=(0, 1))
def train_step(params, opt_state, batch):
    """Single training step with cross-entropy + memory loss."""

    def loss_fn(params):
        # Forward pass
        output = model.apply(
            {'params': params},
            batch['input'],
            return_hidden_states=True,
        )

        # Cross-entropy loss
        xent = cross_entropy_loss(
            output.logits,
            batch['target'],
            batch['loss_mask'],
        )

        # Memory reconstruction loss (on hidden states)
        mem_loss = memory_reconstruction_loss(
            output.hidden_states,
            window_size=EFFECTIVE_WINDOW,
        )

        total_loss = xent + MEMORY_LOSS_WEIGHT * mem_loss
        return total_loss, (xent, mem_loss)

    (loss, (xent, mem_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, {"loss": loss, "xent": xent, "mem_loss": mem_loss}

def batch_examples(examples: list, batch_size: int) -> Iterator[dict]:
    """Batch examples together."""
    np.random.shuffle(examples)
    for i in range(0, len(examples) - batch_size, batch_size):
        batch = examples[i : i + batch_size]
        yield {
            "input": np.stack([e["input"] for e in batch]),
            "target": np.stack([e["target"] for e in batch]),
            "loss_mask": np.stack([e["loss_mask"] for e in batch]),
        }

# === Training Loop ===
NUM_EPOCHS = 1
LOG_EVERY = 10
CHECKPOINT_EVERY = 500  # Save checkpoint every N steps

# Prepare checkpoint directory
import pickle
checkpoint_dir = "/kaggle/working/checkpoint"
os.makedirs(checkpoint_dir, exist_ok=True)

def save_checkpoint(params, step):
    """Save checkpoint with step number."""
    checkpoint_path = os.path.join(checkpoint_dir, f"params_step_{step}.pkl")
    with open(checkpoint_path, "wb") as f:
        pickle.dump(jax.device_get(params), f)
    print(f"Checkpoint saved: {checkpoint_path}")

print("Starting training...")
step = 0

for epoch in range(NUM_EPOCHS):
    for batch in batch_examples(examples, BATCH_SIZE):
        # Convert to JAX arrays
        batch = {k: jnp.array(v) for k, v in batch.items()}

        params, opt_state, metrics = train_step(params, opt_state, batch)

        if step % LOG_EVERY == 0:
            # Convert JAX arrays to Python floats
            loss_val = float(metrics['loss'])
            xent_val = float(metrics['xent'])
            mem_val = float(metrics['mem_loss'])

            # Log to console
            print(f"Step {step}: loss={loss_val:.4f}, xent={xent_val:.4f}, mem_loss={mem_val:.4f}")
            # Log to WandB
            wandb.log({
                "loss": loss_val,
                "xent": xent_val,
                "mem_loss": mem_val,
                "step": step,
            })

        # Periodic checkpoint
        if step > 0 and step % CHECKPOINT_EVERY == 0:
            save_checkpoint(params, step)

        step += 1

print(f"Training complete! Total steps: {step}")

# === Save Checkpoint ===
import pickle
checkpoint_dir = "/kaggle/working/checkpoint"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "memory_trained_params.pkl")
with open(checkpoint_path, "wb") as f:
    pickle.dump(jax.device_get(params), f)  # Move to CPU before saving
print(f"Checkpoint saved to {checkpoint_path}!")

# === Test Generation ===
def generate(prompt: str, max_new_tokens: int = 50):
    """Generate text from prompt."""
    input_ids = tokenizer.encode(prompt)
    input_ids = jnp.array([input_ids])

    for _ in range(max_new_tokens):
        output = model.apply({'params': params}, input_ids)
        next_token = jnp.argmax(output.logits[:, -1, :], axis=-1)
        input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=1)

    return tokenizer.decode(input_ids[0].tolist())

# Test
prompt = "Once upon a time in a distant kingdom,"
print(generate(prompt))

# Finish WandB run
wandb.finish()
