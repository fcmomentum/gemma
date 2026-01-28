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

r"""Example of Gemma fine-tuning with Split-Brain memory heads.

This example shows how to train a Gemma model with the auxiliary memory
reconstruction loss that encourages memory compression.

Training uses two losses:
1. Standard cross-entropy for next-token prediction
2. L1 reconstruction loss for memory heads (predicting T-W state)

Train locally with:

```sh
python -m kauldron.main \
    --cfg=examples/memory_finetuning.py \
    --cfg.workdir=/tmp/memory_training
```

"""

from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from gemma import gm
  from kauldron import kd
  import optax
# pylint: enable=g-import-not-at-top


def get_config():
  batch_size = 16
  max_length = 1024  # Longer sequences to benefit from memory

  return kd.train.Trainer(
      seed=42,
      # Dataset
      train_ds=_make_dataset(
          training=True,
          batch_size=batch_size,
          max_length=max_length,
      ),
      # Model definition - use Memory variant
      model=gm.nn.Gemma3_1B_Memory(
          tokens="batch.input",
          # Memory config with defaults:
          # - memory_head_ratio=0.5 (half of heads for memory)
          # - memory_loss_weight=0.1 (weight of auxiliary loss)
          # - target_layers=None (middle 50% of layers)
      ),
      # Load the weights from the pretrained checkpoint
      # (Same parameters - just different training objective)
      init_transform=gm.ckpts.LoadCheckpoint(
          path=gm.ckpts.CheckpointPath.GEMMA3_1B_IT,
      ),
      # Training
      num_train_steps=10_000,
      train_losses={
          # Standard next-token prediction loss
          "xentropy": kd.losses.SoftmaxCrossEntropyWithIntLabels(
              logits="preds.logits",
              labels="batch.target",
              mask="batch.loss_mask",
          ),
          # Memory reconstruction loss (L1)
          # Trains scribe heads to predict state of token T-W
          "memory": gm.losses.MemoryReconstructionLoss(
              memory_states="preds.memory_states",
              window_size=512,  # Should match model's sliding_window_size
          ),
      },
      optimizer=optax.adafactor(learning_rate=1e-4),
      checkpointer=kd.ckpts.Checkpointer(
          save_interval_steps=500,
      ),
      # Evaluation
      evals={
          "test": kd.evals.Evaluator(
              run=kd.evals.EveryNSteps(1000),
              ds=_make_dataset(
                  training=False,
                  batch_size=batch_size,
                  max_length=max_length,
              ),
          ),
          # Text sampling to verify model still generates coherent text
          "sampling": gm.evals.SamplerEvaluator(
              run=kd.evals.EveryNSteps(1000),
              max_new_tokens=50,
              num_batches=1,
              ds=_make_dataset(training=False, sampling=True),
          ),
      },
  )


def _make_dataset(
    *,
    training: bool,
    sampling: bool = False,
    batch_size: int | None = None,
    max_length: int | None = None,
):
  """Create PG-19 dataset for memory training.

  PG-19 contains full-length books from Project Gutenberg, making it ideal
  for testing long-range memory capabilities.
  """
  tokenizer = gm.text.Gemma3Tokenizer()

  return kd.data.py.Tfds(
      # PG-19: Project Gutenberg books - ideal for long-range memory
      name="pg19",
      split="train" if training else "test",
      shuffle=True if training else False,
      num_epochs=None if training else 1,
      batch_size=None if sampling else batch_size,
      num_workers=4,
      transforms=[
          # For PG-19, we use LM task (next token prediction on long text)
          gm.data.LMTask(
              in_text="book_text",
              out_input="input",
              out_target="target",
              out_target_mask="loss_mask",
              tokenizer=tokenizer,
              max_length=None if sampling else max_length,
              truncate=True,
              sampling=sampling,
          ),
      ],
  )
