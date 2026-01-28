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

"""Memory Reconstruction Loss for Split-Brain attention.

This module implements the L1 reconstruction loss that trains Scribe heads
to predict the latent state of token T-W (the token that just exited the
sliding window).
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import jax
import jax.numpy as jnp
from kauldron import kd
from kauldron import kontext
from kauldron.typing import Float, typechecked


@dataclasses.dataclass(frozen=True, kw_only=True)
class MemoryReconstructionLoss(kd.losses.Loss):
  """L1 reconstruction loss for memory heads.

  This loss trains the Scribe heads to predict the latent state of token T-W
  (the token that just exited the sliding window). By forcing the current
  state to reconstruct the past state, the model learns to compress and
  carry forward information.

  Attributes:
    window_size: The distance between current and target states. Should match
      the model's sliding_window_size so T-W is the token just exiting.
    memory_states: Key to the sowed memory states from attention layers.
      Expected shape: list of [batch, seq, mem_heads, head_dim] tensors.
  """

  window_size: int
  memory_states: kontext.Key = kontext.REQUIRED

  @typechecked
  def get_values(
      self,
      *,
      memory_states: Sequence[Float['B L H D']] | Float['N B L H D'],
  ) -> Float['*B 1']:
    """Computes the L1 reconstruction loss.

    Args:
      memory_states: Memory states sowed from SplitBrainAttention layers.
        Either a list of tensors or a stacked tensor.

    Returns:
      Scalar loss value per batch element.
    """
    # Handle both list and stacked tensor inputs
    if isinstance(memory_states, (list, tuple)):
      stacked_states = jnp.stack(memory_states, axis=0)
    else:
      stacked_states = memory_states

    # stacked_states shape: [num_layers, batch, seq, mem_heads, head_dim]
    num_layers, batch_size, seq_len, mem_heads, head_dim = stacked_states.shape

    # Skip if sequence is too short for reconstruction
    if seq_len <= self.window_size:
      return jnp.zeros((batch_size, 1))

    # Current state (time T): predict from this
    # Positions [window_size, seq_len) correspond to T >= W
    current_states = stacked_states[:, :, self.window_size :, :, :]

    # Target state (time T-W): what we want to predict
    # Positions [0, seq_len - window_size) correspond to T-W
    # Stop gradient on targets - we only want to train current to match past
    target_states = jax.lax.stop_gradient(
        stacked_states[:, :, : seq_len - self.window_size, :, :]
    )

    # L1 loss (Mean Absolute Error)
    l1_loss = jnp.abs(current_states - target_states)

    # Average over layers, sequence, heads, and head_dim
    # Keep batch dimension
    loss_per_batch = jnp.mean(l1_loss, axis=(0, 2, 3, 4))

    return loss_per_batch[:, None]  # [batch, 1] for Loss compatibility


@dataclasses.dataclass(frozen=True, kw_only=True)
class MemoryL1Loss(MemoryReconstructionLoss):
  """Alias for MemoryReconstructionLoss for clarity."""

  pass
