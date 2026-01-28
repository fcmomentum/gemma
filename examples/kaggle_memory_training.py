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
import argparse
from typing import Iterator

# Parse command line arguments
parser = argparse.ArgumentParser(description='Split-Brain Memory Training for Gemma')
parser.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint file to resume from')
parser.add_argument('--max-books', type=int, default=1000,
                    help='Maximum number of books to load (default: 1000)')
parser.add_argument('--max-examples', type=int, default=100000,
                    help='Maximum training examples (default: 100000)')
parser.add_argument('--checkpoint-every', type=int, default=500,
                    help='Save checkpoint every N steps (default: 500)')
parser.add_argument('--no-memory-loss', action='store_true',
                    help='Disable memory loss (for baseline comparison)')
parser.add_argument('--memory-weight', type=float, default=0.1,
                    help='Memory loss weight (default: 0.1)')
args = parser.parse_args()

# Don't force JAX_PLATFORMS - let it auto-detect TPU/GPU/CPU
# The system needs libtpu.so for TPU (only available in system Python, not uv venv)

# Enable JAX compilation cache for faster startup on resume
import os
jax_cache_dir = "/kaggle/working/jax_cache"
os.makedirs(jax_cache_dir, exist_ok=True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = jax_cache_dir

import jax
import jax.numpy as jnp
import optax
import numpy as np
import pickle

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

# Load a subset for training (full dataset is 28K books, 11GB - too slow to tokenize)
train_texts = load_pg19_books("train", max_books=args.max_books)
print(f"Total characters: {sum(len(t) for t in train_texts):,}")

# === Tokenizer and Data Processing ===
from gemma import gm
from tqdm import tqdm

tokenizer = gm.text.Gemma3Tokenizer()  # Use Gemma3 tokenizer

# Configuration for Gemma3 1B
MAX_LENGTH = 1024  # Sequence length (must be > window_size for memory loss)
WINDOW_SIZE = 512   # Gemma3 1B sliding window
BATCH_SIZE = 8      # Larger batch for smaller model

def create_training_examples(texts: list[str], max_length: int, max_examples: int = None) -> Iterator[dict]:
    """Convert book texts to training examples with progress tracking."""
    count = 0
    for text in tqdm(texts, desc="Tokenizing books"):
        # Tokenize
        tokens = tokenizer.encode(text)

        # Create overlapping chunks
        stride = max_length // 2  # 50% overlap
        for i in range(0, len(tokens) - max_length, stride):
            if max_examples and count >= max_examples:
                return
            chunk = tokens[i : i + max_length]
            yield {
                "input": np.array(chunk[:-1], dtype=np.int32),
                "target": np.array(chunk[1:], dtype=np.int32),
                "loss_mask": np.ones(len(chunk) - 1, dtype=np.float32),
            }
            count += 1

# Create examples with progress bar
print(f"Creating training examples (max {args.max_examples:,})...")
examples = list(create_training_examples(train_texts, MAX_LENGTH, args.max_examples))
print(f"Created {len(examples):,} training examples")

# === Load Model and Handle Resume ===
model = gm.nn.Gemma3_1B(tokens="input")

# Check for resume from checkpoint
start_step = 0
if args.resume and os.path.exists(args.resume):
    print(f"Resuming from checkpoint: {args.resume}")
    with open(args.resume, "rb") as f:
        params = pickle.load(f)
    # Extract step number from filename (e.g., params_step_5000.pkl)
    try:
        start_step = int(args.resume.split("_step_")[1].replace(".pkl", ""))
    except:
        start_step = 0
    print(f"Resuming from step {start_step}")
else:
    # Load pretrained weights for training
    # Using PT (pretrained) model for language modeling, not IT (instruction-tuned)
    params = gm.ckpts.load_params(
        path=gm.ckpts.CheckpointPath.GEMMA3_1B_PT,
    )
    print("Loaded pretrained Gemma3 1B PT")

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

# Memory loss weight (0 disables memory loss entirely)
MEMORY_LOSS_WEIGHT = 0.0 if args.no_memory_loss else args.memory_weight
print(f"Memory loss weight: {MEMORY_LOSS_WEIGHT}" + (" (DISABLED)" if args.no_memory_loss else ""))
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
MAX_CHECKPOINTS = 3     # Keep only this many checkpoints (rolling)

# Prepare checkpoint directory
checkpoint_dir = "/kaggle/working/checkpoint"
os.makedirs(checkpoint_dir, exist_ok=True)

def save_checkpoint(params, step):
    """Save checkpoint with rolling deletion of old ones."""
    checkpoint_path = os.path.join(checkpoint_dir, f"params_step_{step}.pkl")
    with open(checkpoint_path, "wb") as f:
        pickle.dump(jax.device_get(params), f)
    print(f"Checkpoint saved: {checkpoint_path}")

    # Rolling checkpoint: delete old checkpoints (sort numerically by step)
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "params_step_*.pkl"))

    def get_step_num(path):
        """Extract step number from checkpoint path."""
        try:
            return int(os.path.basename(path).split("_step_")[1].replace(".pkl", ""))
        except:
            return 0

    # Sort by step number (numerical), not alphabetically
    checkpoints = sorted(checkpoints, key=get_step_num)

    while len(checkpoints) > MAX_CHECKPOINTS:
        old_ckpt = checkpoints.pop(0)
        os.remove(old_ckpt)
        print(f"Deleted old checkpoint: {old_ckpt}")

# === Pre-load Test Set for Periodic Evaluation ===
print("Loading test set for evaluation...")
test_texts = load_pg19_books("test", max_books=50)
test_examples = list(create_training_examples(test_texts, MAX_LENGTH, max_examples=500))
print(f"Test set: {len(test_examples)} examples")

@jax.jit
def compute_loss(params, batch):
    """Compute loss without updating params."""
    output = model.apply(
        {'params': params},
        batch['input'],
        return_hidden_states=True,
    )
    xent = cross_entropy_loss(output.logits, batch['target'], batch['loss_mask'])
    mem_loss = memory_reconstruction_loss(output.hidden_states, window_size=EFFECTIVE_WINDOW)
    return xent, mem_loss

def quick_evaluate(params):
    """Quick evaluation on test set (fewer examples for speed)."""
    total_xent = 0.0
    total_mem_loss = 0.0
    num_batches = 0

    # Use a copy to avoid modifying test_examples
    for batch in batch_examples(test_examples.copy(), BATCH_SIZE):
        batch = {k: jnp.array(v) for k, v in batch.items()}
        xent, mem_loss = compute_loss(params, batch)
        total_xent += float(xent)
        total_mem_loss += float(mem_loss)
        num_batches += 1

    avg_xent = total_xent / max(num_batches, 1)
    avg_mem_loss = total_mem_loss / max(num_batches, 1)
    perplexity = np.exp(avg_xent)

    print(f"  Perplexity: {perplexity:.2f}, XEnt: {avg_xent:.4f}, MemLoss: {avg_mem_loss:.4f}")
    return {"xent": avg_xent, "mem_loss": avg_mem_loss, "perplexity": perplexity}

# === Evaluate Baseline Before Training ===
# At step 0, params == baseline (pretrained model)
if start_step == 0:
    print("\n--- Baseline Evaluation (Before Training) ---")
    baseline_metrics_initial = quick_evaluate(params)
    wandb.log({
        "baseline/xent": baseline_metrics_initial["xent"],
        "baseline/mem_loss": baseline_metrics_initial["mem_loss"],
        "baseline/perplexity": baseline_metrics_initial["perplexity"],
    })
    print(f"Baseline perplexity: {baseline_metrics_initial['perplexity']:.2f}")
else:
    print("\n(Skipping baseline eval - resuming from checkpoint)")

print(f"\nStarting training from step {start_step}...")
step = start_step

for epoch in range(NUM_EPOCHS):
    for batch in batch_examples(examples, BATCH_SIZE):
        # Skip batches if resuming
        if step > start_step:
            pass  # Continue normally

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

        # Periodic checkpoint + evaluation
        if step > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(params, step)

            # Periodic evaluation
            print(f"\n--- Evaluation at step {step} ---")
            eval_metrics = quick_evaluate(params)
            wandb.log({
                "eval/xent": eval_metrics["xent"],
                "eval/mem_loss": eval_metrics["mem_loss"],
                "eval/perplexity": eval_metrics["perplexity"],
                "step": step,
            })

        step += 1

print(f"Training complete! Total steps: {step}")

# === Save Final Checkpoint ===
save_checkpoint(params, step)
print(f"Final checkpoint saved!")

# Reload baseline for final comparison (to save memory during training)
print("\n--- Loading baseline for final comparison ---")
baseline_params = gm.ckpts.load_params(
    path=gm.ckpts.CheckpointPath.GEMMA3_1B_PT,
)

# Evaluate baseline (original Gemma3 1B)
print("\n--- Final Evaluation: Baseline (Original Gemma3 1B PT) ---")
baseline_metrics = quick_evaluate(baseline_params)

# Free baseline params memory
del baseline_params

# Evaluate trained model
print("\n--- Final Evaluation: Trained Model ---")
trained_metrics = quick_evaluate(params)

# Compare results
print("\n" + "="*50)
print("COMPARISON: Baseline vs Trained")
print("="*50)
ppl_delta = trained_metrics["perplexity"] - baseline_metrics["perplexity"]
ppl_improvement = (baseline_metrics["perplexity"] - trained_metrics["perplexity"]) / baseline_metrics["perplexity"] * 100
print(f"  Baseline Perplexity:  {baseline_metrics['perplexity']:.2f}")
print(f"  Trained Perplexity:   {trained_metrics['perplexity']:.2f}")
print(f"  Delta:                {ppl_delta:+.2f}")
print(f"  Improvement:          {ppl_improvement:+.1f}%")

# Log final metrics to WandB
wandb.log({
    "baseline/xent": baseline_metrics["xent"],
    "baseline/mem_loss": baseline_metrics["mem_loss"],
    "baseline/perplexity": baseline_metrics["perplexity"],
    "trained/xent": trained_metrics["xent"],
    "trained/mem_loss": trained_metrics["mem_loss"],
    "trained/perplexity": trained_metrics["perplexity"],
    "comparison/perplexity_delta": ppl_delta,
    "comparison/perplexity_improvement_pct": ppl_improvement,
})

# === Test Generation ===
print("\n=== Sample Generation ===")
def generate(prompt: str, max_new_tokens: int = 100):
    """Generate text from prompt."""
    input_ids = tokenizer.encode(prompt)
    input_ids = jnp.array([input_ids])

    for _ in range(max_new_tokens):
        output = model.apply({'params': params}, input_ids)
        next_token = jnp.argmax(output.logits[:, -1, :], axis=-1)
        input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=1)

    return tokenizer.decode(input_ids[0].tolist())

# Test with a book-style prompt
prompt = "Once upon a time in a distant kingdom, there lived a young prince named"
print(f"Prompt: {prompt}")
print(f"Generated: {generate(prompt)}")

# Finish WandB run
wandb.finish()
print("\nDone!")
