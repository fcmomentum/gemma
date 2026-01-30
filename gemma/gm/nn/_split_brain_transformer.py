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

"""Split-Brain Transformer model.

A Transformer variant that uses Split-Brain attention at specified layers
to improve reasoning and planning capabilities.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import flax
from flax import linen as nn
from gemma.gm.nn import _config
from gemma.gm.nn import _layers
from gemma.gm.nn import _modules
from gemma.gm.nn import _split_brain
from gemma.gm.nn import _transformer
from gemma.gm.utils import _dtype_params
from gemma.gm.utils import _jax_utils
import jax.numpy as jnp
from kauldron.typing import Bool, Float, Int, typechecked


@flax.struct.dataclass
class SplitBrainOutput:
  """Output of the Split-Brain Gemma model.

  Attributes:
    logits: Predicted logits of the model.
    cache: Updated cache if the input cache is not None, None elsewhere.
    hidden_states: The hidden states of the model.
    aux_loss: Auxiliary prophet loss from Split-Brain layers.
  """

  logits: Float['*B L V'] | Float['*B V']
  cache: _config.Cache | None
  hidden_states: Float['*B L D'] | Float['*B D'] | None
  aux_loss: Float[''] | None  # Scalar auxiliary loss


class SplitBrainTransformer(nn.Module):
  """Transformer with Split-Brain attention at specified layers.

  This extends the base Transformer to replace certain layers with
  SplitBrainBlock, enabling the "Prophet" training objective.

  Attributes:
    config: Standard TransformerConfig.
    split_brain_config: Configuration for Split-Brain layers.
    return_last_only: If True, only return last token logits.
    dtype: Parameter dtype (default bfloat16).
  """

  config: _config.TransformerConfig
  split_brain_config: _split_brain.SplitBrainConfig

  _: dataclasses.KW_ONLY
  return_last_only: bool | None = None
  dtype: jnp.dtype = jnp.bfloat16

  def setup(self):
    self.embedder = _modules.Embedder(
        vocab_size=self.config.num_embed,
        embed_dim=self.config.embed_dim,
        vision_proj_dim=self.config.vision_encoder.siglip_encoder.width
        if self.config.vision_encoder
        else None,
    )

    # Build blocks - use SplitBrainBlock at specified layer indices
    split_layers = set(self.split_brain_config.split_brain_layers)
    blocks = []
    split_brain_indices = []

    for i, attn_type in enumerate(self.config.attention_types):
      if i in split_layers:
        # Use Split-Brain block
        block = _split_brain.SplitBrainBlock(
            name=f'layer_{i}',
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            embed_dim=self.config.embed_dim,
            head_dim=self.config.head_dim,
            hidden_dim=self.config.hidden_dim,
            sliding_window_size=self.config.sliding_window_size,
            use_post_attn_norm=self.config.use_post_attn_norm,
            use_post_ffw_norm=self.config.use_post_ffw_norm,
            attn_logits_soft_cap=self.config.attn_logits_soft_cap,
            attn_type=attn_type,
            query_pre_attn_scalar=self.config.query_pre_attn_scalar(),
            transpose_gating_einsum=self.config.transpose_gating_einsum,
            use_qk_norm=self.config.use_qk_norm,
            split_brain_config=self.split_brain_config,
            rope_base_frequency=self.config.local_base_frequency
            if attn_type == _modules.AttentionType.LOCAL_SLIDING
            else self.config.global_base_frequency,
            rope_scale_factor=self.config.local_scale_factor
            if attn_type == _modules.AttentionType.LOCAL_SLIDING
            else self.config.global_scale_factor,
        )
        split_brain_indices.append(i)
      else:
        # Use standard block
        block = _modules.Block(
            name=f'layer_{i}',
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            embed_dim=self.config.embed_dim,
            head_dim=self.config.head_dim,
            hidden_dim=self.config.hidden_dim,
            sliding_window_size=self.config.sliding_window_size,
            use_post_attn_norm=self.config.use_post_attn_norm,
            use_post_ffw_norm=self.config.use_post_ffw_norm,
            attn_logits_soft_cap=self.config.attn_logits_soft_cap,
            attn_type=attn_type,
            query_pre_attn_scalar=self.config.query_pre_attn_scalar(),
            transpose_gating_einsum=self.config.transpose_gating_einsum,
            use_qk_norm=self.config.use_qk_norm,
            rope_base_frequency=self.config.local_base_frequency
            if attn_type == _modules.AttentionType.LOCAL_SLIDING
            else self.config.global_base_frequency,
            rope_scale_factor=self.config.local_scale_factor
            if attn_type == _modules.AttentionType.LOCAL_SLIDING
            else self.config.global_scale_factor,
        )
      blocks.append(block)

    self.blocks = blocks
    self.split_brain_indices = split_brain_indices

    self.final_norm = _layers.RMSNorm()

  @functools.partial(
      nn.jit,
      static_argnames=(
          'self',
          'return_last_only',
          'return_hidden_states',
          'deterministic',
      ),
  )
  @_jax_utils.flatten_unflatten_batch_dim()
  @typechecked
  def __call__(
      self,
      tokens: Int['*B L'],
      *,
      positions: Int['*B L'] | None = None,
      cache: _config.Cache | None = None,
      attention_mask: Bool['*B L cache_length'] | None = None,
      return_last_only: bool | None = None,
      return_hidden_states: bool | None = None,
      deterministic: bool = False,
  ) -> SplitBrainOutput:
    """Forward pass with Split-Brain attention.

    Args:
      tokens: Input token IDs [B, L].
      positions: Optional absolute positions [B, L].
      cache: KV cache or None.
      attention_mask: Attention mask [B, L, cache_len].
      return_last_only: If True, only return last token logits.
      return_hidden_states: If True, return hidden states.
      deterministic: If True, disable random masking (for eval/inference).

    Returns:
      SplitBrainOutput with logits, cache, hidden_states, and aux_loss.
    """
    if return_last_only is None:
      return_last_only = self.return_last_only or False

    with _dtype_params.initialize_param_with_dtype(
        self.dtype,
        exclude=['lora'],
    ):
      # Encode tokens
      x = self.embedder.encode(tokens)

      # Build positions if not provided
      if positions is None:
        batch_size, seq_len = tokens.shape
        positions = jnp.broadcast_to(
            jnp.arange(seq_len), (batch_size, seq_len)
        )

      # Build attention mask if not provided
      if attention_mask is None:
        seq_len = tokens.shape[1]
        cache_len = cache['layer_0']['v'].shape[1] if cache else seq_len
        # Causal mask
        attention_mask = jnp.tril(
            jnp.ones((seq_len, cache_len), dtype=jnp.bool_)
        )
        attention_mask = jnp.broadcast_to(
            attention_mask, (tokens.shape[0], seq_len, cache_len)
        )

      # Apply transformer blocks
      x, new_cache, aux_loss = self._apply_attention(
          x, positions, attention_mask, cache, deterministic
      )

    if return_last_only:
      x = x[:, -1, :]

    logits = self.embedder.decode(x)

    if self.config.final_logit_softcap is not None:
      logits /= self.config.final_logit_softcap
      logits = jnp.tanh(logits) * self.config.final_logit_softcap

    return SplitBrainOutput(
        logits=logits,
        cache=None if cache is None else new_cache,
        hidden_states=x if return_hidden_states else None,
        aux_loss=aux_loss,
    )

  def _apply_attention(
      self,
      x: Float['B L D'],
      positions: Int['B L'],
      attention_mask: Bool['B L cache_len'],
      cache: _config.Cache | None,
      deterministic: bool,
  ) -> tuple[Float['B L D'], _config.Cache, Float['']]:
    """Apply transformer blocks and collect auxiliary losses."""
    old_cache = cache or {}
    new_cache = {}
    aux_losses = []

    split_layers = set(self.split_brain_indices)

    for i, block in enumerate(self.blocks):
      layer_name = f'layer_{i}'

      if i in split_layers:
        # SplitBrainBlock returns (cache, output, aux_loss)
        layer_cache, x, layer_aux_loss = block(
            x,
            positions,
            old_cache.get(layer_name),
            attention_mask,
            deterministic=deterministic,
        )
        aux_losses.append(layer_aux_loss)
      else:
        # Standard Block returns (cache, output)
        layer_cache, x = block(
            x,
            positions,
            old_cache.get(layer_name),
            attention_mask,
        )

      new_cache[layer_name] = layer_cache

    x = self.final_norm(x)

    # Sum auxiliary losses from all Split-Brain layers
    if aux_losses:
      # Stack: [num_layers, B]
      # Sum over layers to get [B]
      total_aux_loss = jnp.sum(jnp.stack(aux_losses), axis=0)
    else:
      # Return zero array with batch size
      batch_size = x.shape[0]
      total_aux_loss = jnp.zeros((batch_size,), dtype=jnp.float32)

    return x, new_cache, total_aux_loss

  def init_cache(
      self,
      *,
      batch_size: int,
      dtype: jnp.dtype[Any],
      cache_length: int,
  ) -> _config.Cache:
    """Initialize KV cache."""
    return self.config.init_cache(
        batch_size=batch_size,
        dtype=dtype,
        cache_length=cache_length,
    )
