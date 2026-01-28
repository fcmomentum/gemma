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

"""Memory Transformer with Split-Brain attention heads.

This module extends the standard Transformer to use SplitBrainBlock for
target layers, enabling memory training with auxiliary reconstruction loss.
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
from gemma.gm.nn import _transformer
from gemma.gm.nn._memory_attention import MemoryConfig
from gemma.gm.nn._memory_attention import SplitBrainBlock
from gemma.gm.utils import _dtype_params
from gemma.gm.utils import _jax_utils
import jax.numpy as jnp
from kauldron import kd
from kauldron.typing import Bool, Float, Int, UInt8, typechecked


@flax.struct.dataclass
class MemoryOutput:
  """Output of the MemoryTransformer.

  Attributes:
    logits: Predicted logits of the model.
    cache: Updated cache if the input cache is not None, None elsewhere.
    hidden_states: The hidden states of the model.
    memory_states: Stacked memory states from SplitBrain layers for loss.
  """

  logits: Float['*B L V'] | Float['*B V']
  cache: _config.Cache | None
  hidden_states: Float['*B L D'] | Float['*B D'] | None
  memory_states: Float['N B L H D'] | None


class MemoryTransformer(_transformer.Transformer):
  """Transformer with Split-Brain memory heads.

  This extends the standard Transformer to use SplitBrainBlock for designated
  layers, enabling training with the auxiliary memory reconstruction loss.

  Attributes:
    memory_config: Configuration for Split-Brain memory heads.
    enable_memory: Whether to enable memory sowing (disable during inference).
  """

  memory_config: MemoryConfig = MemoryConfig()
  enable_memory: bool = True

  def setup(self):
    # Determine which layers should use memory attention
    memory_layer_indices = set(
        self.memory_config.get_target_layers(self.config.num_layers)
    )

    # Build blocks - use SplitBrainBlock for memory layers
    self.blocks = []
    for i, attn_type in enumerate(self.config.attention_types):
      is_memory_layer = i in memory_layer_indices

      rope_base_frequency = (
          self.config.local_base_frequency
          if attn_type == _modules.AttentionType.LOCAL_SLIDING
          else self.config.global_base_frequency
      )
      rope_scale_factor = (
          self.config.local_scale_factor
          if attn_type == _modules.AttentionType.LOCAL_SLIDING
          else self.config.global_scale_factor
      )

      if is_memory_layer and self.enable_memory:
        block = SplitBrainBlock(
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
            rope_base_frequency=rope_base_frequency,
            rope_scale_factor=rope_scale_factor,
            memory_config=self.memory_config,
            enable_memory_sow=self.enable_memory,
        )
      else:
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
            rope_base_frequency=rope_base_frequency,
            rope_scale_factor=rope_scale_factor,
        )
      self.blocks.append(block)

    # Standard embedder and final norm
    self.embedder = _modules.Embedder(
        vocab_size=self.config.num_embed,
        embed_dim=self.config.embed_dim,
        vision_proj_dim=self.config.vision_encoder.siglip_encoder.width
        if self.config.vision_encoder
        else None,
    )
    self.final_norm = _layers.RMSNorm()
    self.vision_encoder = self.config.vision_encoder

  @functools.partial(
      nn.jit,
      static_argnames=(
          'self',
          'return_last_only',
          'return_hidden_states',
          'return_memory_states',
      ),
  )
  @_jax_utils.flatten_unflatten_batch_dim()
  @typechecked
  def __call__(
      self,
      tokens: Int['*B L'],
      *,
      images: UInt8['*B N H W C'] | UInt8['*B H W C'] | None = None,
      positions: Int['*B L_with_mm'] | None = None,
      cache: _config.Cache | None = None,
      attention_mask: Bool['*B L_with_mm cache_length'] | None = None,
      return_last_only: bool | None = None,
      return_hidden_states: bool | None = None,
      return_memory_states: bool = True,
  ) -> MemoryOutput:
    """Transformer forward pass with memory state collection.

    Args:
      tokens: input sequence of tokens.
      images: Images to feed to the vision encoder.
      positions: input absolute positions.
      cache: Attention KV cache or None.
      attention_mask: transformer input mask.
      return_last_only: If True, only compute and return the last token logits.
      return_hidden_states: If True, return hidden states.
      return_memory_states: If True, return stacked memory states for loss.

    Returns:
      MemoryOutput with logits, cache, hidden_states, and memory_states.
    """
    return_last_only = self._get_return_last_only(return_last_only)

    with _dtype_params.initialize_param_with_dtype(
        self.dtype,
        exclude=[
            'vision_encoder',
            'embedder.mm_input_projection',
            'embedder.mm_soft_embedding_norm',
            'lora',
        ],
    ):
      inputs = self._encode_and_get_inputs(
          tokens=tokens,
          images=images,
          positions=positions,
          attention_mask=attention_mask,
      )
      del positions, attention_mask

      # Run attention with intermediate capture for memory states
      if self.enable_memory and return_memory_states:
        x, new_cache, memory_states = self._apply_attention_with_memory(
            inputs, cache
        )
      else:
        x, new_cache = self._apply_attention(inputs, cache)
        memory_states = None

    if return_last_only:
      last_input_token_idx = jnp.sum(inputs.inputs_mask, axis=-1) - 1
      x = x[jnp.arange(len(x)), last_input_token_idx, ...]

    logits = self.embedder.decode(x)

    if self.config.final_logit_softcap is not None:
      logits /= self.config.final_logit_softcap
      logits = jnp.tanh(logits) * self.config.final_logit_softcap

    return MemoryOutput(
        logits=logits,
        cache=None if cache is None else new_cache,
        hidden_states=x if return_hidden_states else None,
        memory_states=memory_states,
    )

  def _apply_attention_with_memory(
      self, inputs: _transformer._Inputs, cache: _config.Cache | None
  ) -> tuple[Float['*B L D'], _config.Cache, Float['N B L H D'] | None]:
    """Runs transformer blocks and collects memory states.

    Uses capture_intermediates to gather the sowed memory_state tensors
    from SplitBrainAttention layers.
    """
    x = inputs.embeddings
    old_cache = cache or {}
    new_cache = {}
    memory_states = []

    for i, block in enumerate(self.blocks):
      layer_name = f'layer_{i}'

      if isinstance(block, SplitBrainBlock):
        # For SplitBrain blocks, capture intermediates
        (layer_cache, x), variables = block.apply(
            {'params': block.variables['params']},
            x,
            inputs.positions,
            old_cache.get(layer_name),
            inputs.attention_mask,
            capture_intermediates=True,
            mutable=['intermediates'],
        )

        # Extract sowed memory states
        if 'intermediates' in variables:
          intermediates = variables['intermediates']
          if 'attn' in intermediates and 'memory_state' in intermediates['attn']:
            mem_state = intermediates['attn']['memory_state']['__call__'][0]
            memory_states.append(mem_state)
      else:
        layer_cache, x = block(
            x,
            inputs.positions,
            old_cache.get(layer_name),
            inputs.attention_mask,
        )

      new_cache[layer_name] = layer_cache

    x = self.final_norm(x)

    # Stack memory states: [num_memory_layers, batch, seq, heads, dim]
    if memory_states:
      stacked_memory = jnp.stack(memory_states, axis=0)
    else:
      stacked_memory = None

    return x, new_cache, stacked_memory
