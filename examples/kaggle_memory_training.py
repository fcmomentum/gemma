# Split-Brain Memory Training on Kaggle
# =====================================
# This notebook trains Gemma with Split-Brain memory heads using PG-19 dataset.
#
# Setup:
# 1. Add Kaggle dataset: "the-pg19-language-modeling-benchmark-dataset"
# 2. Enable TPU accelerator
# 3. Run all cells

# %% [markdown]
# ## Cell 1: Install Dependencies

# %%
# !pip install -q gemma flax optax jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# %% [markdown]
# ## Cell 2: Imports and Device Check

# %%
import os
import glob
import functools
from typing import Iterator

import jax
import jax.numpy as jnp
import optax
import numpy as np

print(f"JAX devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")

# %% [markdown]
# ## Cell 3: Load PG-19 Dataset from Kaggle

# %%
PG19_PATH = "/kaggle/input/the-pg19-language-modeling-benchmark-dataset"

def load_pg19_books(split: str = "train", max_books: int = None) -> list[str]:
    """Load book texts from PG-19 dataset."""
    # Try common directory structures
    possible_paths = [
        os.path.join(PG19_PATH, split),
        os.path.join(PG19_PATH, f"{split}.txt"),
        PG19_PATH,
    ]

    texts = []
    for path in possible_paths:
        if os.path.isdir(path):
            files = glob.glob(os.path.join(path, "*.txt"))
            for f in files[:max_books] if max_books else files:
                with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
                    texts.append(fp.read())
            break
        elif os.path.isfile(path):
            with open(path, 'r', encoding='utf-8', errors='ignore') as fp:
                texts.append(fp.read())
            break

    print(f"Loaded {len(texts)} books from {split}")
    return texts

# Load a subset for testing
train_texts = load_pg19_books("train", max_books=100)  # Start small
print(f"Total characters: {sum(len(t) for t in train_texts):,}")

# %% [markdown]
# ## Cell 4: Tokenizer and Data Processing

# %%
from gemma import gm

tokenizer = gm.text.Gemma2Tokenizer()  # Use Gemma2 tokenizer

# Configuration
MAX_LENGTH = 2048  # Sequence length (adjust based on TPU memory)
WINDOW_SIZE = 4096  # Gemma2 sliding window (for reference)
BATCH_SIZE = 4      # Small batch for TPU memory

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

# %% [markdown]
# ## Cell 5: Load Split-Brain Memory Model

# %%
# Import the memory modules (copy these files to Kaggle or install gemma with memory support)
# For now, we'll use standard Gemma2 and add memory loss manually

model = gm.nn.Gemma2_2B(tokens="input")

# Load pretrained weights
params = gm.ckpts.load_params(
    path=gm.ckpts.CheckpointPath.GEMMA2_2B_IT,
    model=model,
)
print("Model loaded successfully!")

# %% [markdown]
# ## Cell 6: Define Loss Functions

# %%
def cross_entropy_loss(logits, targets, mask):
    """Standard next-token prediction loss."""
    # Shift for causal LM
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
        return 0.0

    seq_len = hidden_states.shape[1]
    if seq_len <= window_size:
        return 0.0

    # Current state (time T)
    current = hidden_states[:, window_size:, :]
    # Target state (time T-W) - stop gradient!
    target = jax.lax.stop_gradient(hidden_states[:, :-window_size, :])

    # L1 loss
    return jnp.mean(jnp.abs(current - target))

# %% [markdown]
# ## Cell 7: Training Step

# %%
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
        # Note: For full Split-Brain, you'd use memory_states from SplitBrainAttention
        # Here we approximate using final hidden states
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

# %% [markdown]
# ## Cell 8: Training Loop

# %%
NUM_EPOCHS = 1
LOG_EVERY = 10

print("Starting training...")
step = 0

for epoch in range(NUM_EPOCHS):
    for batch in batch_examples(examples, BATCH_SIZE):
        # Convert to JAX arrays
        batch = {k: jnp.array(v) for k, v in batch.items()}

        params, opt_state, metrics = train_step(params, opt_state, batch)

        if step % LOG_EVERY == 0:
            print(f"Step {step}: loss={metrics['loss']:.4f}, "
                  f"xent={metrics['xent']:.4f}, mem_loss={metrics['mem_loss']:.4f}")

        step += 1

print(f"Training complete! Total steps: {step}")

# %% [markdown]
# ## Cell 9: Save Checkpoint

# %%
# Save trained params
import pickle
with open("/kaggle/working/memory_trained_params.pkl", "wb") as f:
    pickle.dump(params, f)
print("Checkpoint saved!")

# %% [markdown]
# ## Cell 10: Test Generation

# %%
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
