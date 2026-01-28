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

"""Split-Brain Memory Attention module.

This module implements the Split-Brain architecture that partitions attention
heads into Navigator (standard prediction) and Scribe (memory compression) heads.
Both head types use sliding window attention, but Scribe heads are trained with
an auxiliary L1 reconstruction loss to predict the state of token T-W (the token
that just exited the sliding window).
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

from flax import linen as nn
from gemma.gm.math import _positional_embeddings
from gemma.gm.nn import _layers
from gemma.gm.nn import _modules
import jax
import jax.numpy as jnp
from kauldron.typing import Bool, Int  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(frozen=True)
class MemoryConfig:
  """Configuration for Split-Brain memory heads.

  Attributes:
    memory_head_ratio: Ratio of heads dedicated to memory (0.0 to 1.0).
      Default is 0.5 (half of heads).
    memory_window_size: Window size for reconstruction loss. If None, uses
      the model's sliding_window_size.
    memory_loss_weight: Weight of the auxiliary reconstruction loss.
    target_layers: List of layer indices to apply memory loss. If None,
      defaults to the middle 50% of layers.
  """

  memory_head_ratio: float = 0.5
  memory_window_size: int | None = None
  memory_loss_weight: float = 0.1
  target_layers: Sequence[int] | None = None

  def get_num_memory_heads(self, num_heads: int) -> int:
    """Calculate the number of memory heads based on ratio."""
    return max(1, int(num_heads * self.memory_head_ratio))

  def get_target_layers(self, num_layers: int) -> Sequence[int]:
    """Get the list of layers that should use memory heads."""
    if self.target_layers is not None:
      return self.target_layers
    # Default: middle 50% of layers
    start = num_layers // 4
    end = num_layers - (num_layers // 4)
    return list(range(start, end))


class SplitBrainAttention(nn.Module):
  """Attention module with Split-Brain memory heads.

  This extends the standard Attention to partition heads into Navigator and
  Scribe groups. Both use the same sliding window attention, but Scribe head
  outputs are captured via sow() for the auxiliary reconstruction loss.

  Attributes:
    num_heads: Total number of attention heads.
    num_kv_heads: Number of key-value heads (for GQA).
    features: Feature dimension (embed_dim).
    head_dim: Dimension per head.
    attn_type: Attention type (GLOBAL or LOCAL_SLIDING).
    query_pre_attn_scalar: Scaling factor for query.
    memory_config: Configuration for memory heads.
    enable_memory_sow: Whether to sow memory states (disable during inference).
  """

  num_heads: int
  num_kv_heads: int
  features: int
  head_dim: int
  attn_type: _modules.AttentionType
  query_pre_attn_scalar: float
  memory_config: MemoryConfig
  rope_base_frequency: int = _modules.DEFAULT_ROPE_BASE_FREQUENCY
  rope_scale_factor: float = _modules.DEFAULT_ROPE_SCALE_FACTOR
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  use_qk_norm: bool = False
  enable_memory_sow: bool = True

  @property
  def num_memory_heads(self) -> int:
    """Number of heads dedicated to memory."""
    return self.memory_config.get_num_memory_heads(self.num_heads)

  @property
  def num_navigator_heads(self) -> int:
    """Number of heads for standard prediction."""
    return self.num_heads - self.num_memory_heads

  @property
  def use_qkv_einsum(self):
    return self.num_kv_heads == self.num_heads

  @property
  def use_gqa(self):
    return self.num_kv_heads != self.num_heads and self.num_kv_heads > 1

  def setup(self):
    self.attn_vec_einsum = _layers.Einsum(
        shape=(self.num_heads, self.head_dim, self.features),
    )

    if self.use_qkv_einsum:
      self.qkv_einsum = _layers.Einsum(
          shape=(3, self.num_heads, self.features, self.head_dim),
      )
    else:
      self.q_einsum = _layers.Einsum(
          shape=(self.num_heads, self.features, self.head_dim),
      )
      self.kv_einsum = _layers.Einsum(
          shape=(2, self.num_kv_heads, self.features, self.head_dim),
      )
    if self.use_qk_norm:
      self._query_norm = _layers.RMSNorm()
      self._key_norm = _layers.RMSNorm()

    self.attention_weights = nn.Identity()

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: _modules.LayerCache | None,
      attn_mask: jax.Array,
  ) -> tuple[_modules.LayerCache | None, jax.Array]:
    """Applies Split-Brain attention to the inputs.

    Args:
      x: Input sequence of shape [batch_size, seq_len, embed_dim].
      segment_pos: Input absolute positions of shape [batch_size, seq_len].
      cache: KV cache or None.
      attn_mask: Attention mask of shape [batch_size, seq_len, cache_size].

    Returns:
      cache: Updated attention KV cache.
      outputs: Output sequence of shape [batch_size, seq_len, embed_dim].
    """
    if self.use_qkv_einsum:
      query_proj, key_proj, value_proj = self.qkv_einsum('BTD,SNDH->SBTNH', x)
    else:
      query_proj = self.q_einsum('BTD,NDH->BTNH', x)
      key_proj, value_proj = self.kv_einsum('BSD,CKDH->CBSKH', x)

    if self.use_qk_norm:
      query_proj = self._query_norm(query_proj)
      key_proj = self._key_norm(key_proj)

    query_proj = _positional_embeddings.apply_rope(
        query_proj,
        segment_pos,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
    )
    query_scaled = query_proj * self.query_pre_attn_scalar

    key_proj = _positional_embeddings.apply_rope(
        key_proj,
        segment_pos,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
    )

    # Cache handling (same as standard attention)
    if cache is not None:
      end_index = cache['end_index'][0]
      cache_size = cache['v'].shape[1]
      update_index = end_index % cache_size
      slice_indices = (0, update_index, 0, 0)

      value_proj = jax.lax.dynamic_update_slice(
          cache['v'],
          value_proj,
          slice_indices,
      )

      key_proj = jax.lax.dynamic_update_slice(
          cache['k'],
          key_proj,
          slice_indices,
      )

      cache_positions = jax.lax.dynamic_update_slice(
          cache['positions'],
          segment_pos,
          slice_indices[:2],
      )

    # Compute attention (both Navigator and Scribe use same sliding window)
    if self.use_gqa:
      b, t, kg, h = query_scaled.shape
      query_scaled = query_scaled.reshape(
          (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
      )
      logits = jnp.einsum('BTKGH,BSKH->BTKGS', query_scaled, key_proj)
      b, t, k, g, s = logits.shape
      logits = logits.reshape((b, t, k * g, s))
    else:
      logits = jnp.einsum('BTNH,BSNH->BTNS', query_scaled, key_proj)

    if self.attn_logits_soft_cap is not None:
      logits = jnp.tanh(logits / self.attn_logits_soft_cap)
      logits = logits * self.attn_logits_soft_cap

    # Apply sliding window mask (same for both head types)
    if self.attn_type == _modules.AttentionType.LOCAL_SLIDING:
      if self.sliding_window_size is None:
        raise ValueError(
            'Sliding_window_size must be set if Local Sliding attention type'
        )
      sliding_mask = _modules.create_sliding_mask(
          segment_pos,
          cache_positions=cache_positions if cache else None,  # pylint: disable=undefined-variable
          sliding_window_size=self.sliding_window_size,
      )
      attn_mask = attn_mask * sliding_mask

    padded_logits = jnp.where(
        (jnp.expand_dims(attn_mask, -2)), logits, _modules.K_MASK
    )

    probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_proj.dtype)
    probs = self.attention_weights(probs)

    if self.use_gqa:
      b, t, kg, h = probs.shape
      probs = probs.reshape(
          (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
      )
      encoded = jnp.einsum('BTKGS,BSKH->BTKGH', probs, value_proj)
      b, t, k, g, h = encoded.shape
      encoded = encoded.reshape((b, t, k * g, h))
    else:
      encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)

    # Split heads into Navigator and Scribe
    # encoded shape: [batch_size, seq_len, num_heads, head_dim]
    navigator_encoded = encoded[:, :, : self.num_navigator_heads, :]
    scribe_encoded = encoded[:, :, self.num_navigator_heads :, :]

    # Sow the Scribe head outputs for the reconstruction loss
    # Only sow during training (enable_memory_sow=True)
    if self.enable_memory_sow:
      self.sow('intermediates', 'memory_state', scribe_encoded)

    # Recombine for output projection (both contribute to output)
    combined_encoded = jnp.concatenate(
        [navigator_encoded, scribe_encoded], axis=2
    )

    attn_output = self.attn_vec_einsum('BTNH,NHD->BTD', combined_encoded)

    if cache is not None:
      seq_len = x.shape[1]
      new_cache = {
          'v': value_proj,
          'k': key_proj,
          'end_index': cache['end_index'] + seq_len,
          'positions': cache_positions,  # pylint: disable=undefined-variable
      }
    else:
      new_cache = None

    return new_cache, attn_output

  @classmethod
  def init_cache(
      cls,
      cache_size: int,
      num_heads: int,
      head_dim: int,
      batch_size: int,
      dtype: jnp.dtype = jnp.bfloat16,
  ) -> _modules.LayerCache:
    """Initialize KV cache (same as standard Attention)."""
    return _modules.Attention.init_cache(
        cache_size=cache_size,
        num_heads=num_heads,
        head_dim=head_dim,
        batch_size=batch_size,
        dtype=dtype,
    )


class SplitBrainBlock(nn.Module):
  """Transformer block with Split-Brain attention.

  This wraps the standard Block to use SplitBrainAttention for layers that
  are configured for memory training.
  """

  num_heads: int
  num_kv_heads: int
  embed_dim: int
  head_dim: int
  hidden_dim: int
  use_post_attn_norm: bool
  use_post_ffw_norm: bool
  attn_type: _modules.AttentionType
  query_pre_attn_scalar: float
  transpose_gating_einsum: bool
  memory_config: MemoryConfig
  rope_base_frequency: int = _modules.DEFAULT_ROPE_BASE_FREQUENCY
  rope_scale_factor: float = _modules.DEFAULT_ROPE_SCALE_FACTOR
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  use_qk_norm: bool = False
  enable_memory_sow: bool = True

  def setup(self):
    self.pre_attention_norm = _layers.RMSNorm()

    self.attn = SplitBrainAttention(
        num_heads=self.num_heads,
        features=self.embed_dim,
        head_dim=self.head_dim,
        num_kv_heads=self.num_kv_heads,
        attn_type=self.attn_type,
        query_pre_attn_scalar=self.query_pre_attn_scalar,
        memory_config=self.memory_config,
        rope_base_frequency=self.rope_base_frequency,
        rope_scale_factor=self.rope_scale_factor,
        attn_logits_soft_cap=self.attn_logits_soft_cap,
        sliding_window_size=self.sliding_window_size,
        use_qk_norm=self.use_qk_norm,
        enable_memory_sow=self.enable_memory_sow,
    )

    self.post_attention_norm = None
    if self.use_post_attn_norm:
      self.post_attention_norm = _layers.RMSNorm()

    self.pre_ffw_norm = _layers.RMSNorm()

    self.mlp = _modules.FeedForward(
        features=self.embed_dim,
        hidden_dim=self.hidden_dim,
        transpose_gating_einsum=self.transpose_gating_einsum,
    )

    self.post_ffw_norm = None
    if self.use_post_ffw_norm:
      self.post_ffw_norm = _layers.RMSNorm()

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: _modules.LayerCache | None,
      attn_mask: jax.Array,
  ) -> tuple[_modules.LayerCache | None, jax.Array]:
    """Applies the block to the inputs.

    Args:
      x: Input sequence of shape [batch_size, seq_len, embed_dim].
      segment_pos: Input absolute positions of shape [batch_size, seq_len].
      cache: KV cache or None.
      attn_mask: Attention mask of shape [batch_size, seq_len, cache_size].

    Returns:
      cache: Updated attention KV cache.
      outputs: Output sequence of shape [batch_size, seq_len, embed_dim].
    """
    inputs_normalized = self.pre_attention_norm(x)

    cache, attn_output = self.attn(
        inputs_normalized,
        segment_pos,
        cache,
        attn_mask,
    )

    if self.post_attention_norm is not None:
      attn_output = self.post_attention_norm(attn_output)

    attn_output += x

    outputs = self.pre_ffw_norm(attn_output)

    outputs = self.mlp(outputs)

    if self.post_ffw_norm is not None:
      outputs = self.post_ffw_norm(outputs)

    outputs += attn_output

    return cache, outputs
