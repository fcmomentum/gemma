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

"""Split-Brain Prophet attention modules.

The Split-Brain architecture introduces a dual-stream attention mechanism:
- Teacher Stream: Standard causal attention (the "control")
- Student Stream: Masked causal attention trained to predict Teacher's future states

This forces the model to learn robust global representations and improve planning.
"""

import dataclasses

from flax import linen as nn
from gemma.gm.nn import _layers
from gemma.gm.nn import _modules
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class SplitBrainConfig:
  """Configuration for Split-Brain Prophet mechanism."""

  mask_ratio: float = 0.5  # Probability of masking tokens in Student stream
  prophet_weight: float = 0.1  # λ weight for auxiliary loss
  target_shift: int = 1  # Student(t) predicts Teacher(t+1)
  stop_gradient: bool = True  # Detach Teacher target from gradient
  gate_init_bias: float = 2.0  # Gate bias init (sigmoid(2.0) ≈ 0.88)
  split_brain_layers: tuple[int, ...] = ()  # Layer indices to apply Split-Brain


class GatedFusion(nn.Module):
  """Gated fusion mechanism for combining Teacher and Student outputs.

  Uses a learned gate that is initialized to prioritize the Teacher stream
  at the start of training (bias initialized to +2.0 → sigmoid ≈ 0.88).
  """

  features: int
  gate_init_bias: float = 2.0

  @nn.compact
  def __call__(
      self,
      teacher_out: jax.Array,
      student_out: jax.Array,
  ) -> jax.Array:
    """Apply gated fusion.

    Args:
      teacher_out: Output from Teacher stream [B, L, D].
      student_out: Output from Student stream [B, L, D].

    Returns:
      Fused output [B, L, D].
    """
    # Concatenate teacher and student outputs
    combined = jnp.concatenate([teacher_out, student_out], axis=-1)

    # Linear projection to gate dimension
    gate_proj = nn.Dense(
        features=self.features,
        kernel_init=nn.initializers.normal(stddev=0.02),
        bias_init=nn.initializers.constant(self.gate_init_bias),
        name='gate_projection',
    )(combined)

    # Sigmoid gate: at init, sigmoid(2.0) ≈ 0.88 prioritizes Teacher
    gate = jax.nn.sigmoid(gate_proj)

    # Blend outputs: high gate = more Teacher, low gate = more Student
    output = gate * teacher_out + (1.0 - gate) * student_out

    return output


class SplitBrainAttention(nn.Module):
  """Split-Brain dual-stream attention module.

  Contains two parallel attention streams:
  - Teacher: Standard causal attention
  - Student: Causal + random token masking, trained to predict Teacher(t+1)
  """

  num_heads: int
  num_kv_heads: int
  features: int
  head_dim: int
  attn_type: _modules.AttentionType
  query_pre_attn_scalar: float
  mask_ratio: float = 0.15
  rope_base_frequency: int = _modules.DEFAULT_ROPE_BASE_FREQUENCY
  rope_scale_factor: float = _modules.DEFAULT_ROPE_SCALE_FACTOR
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  use_qk_norm: bool = False

  def setup(self):
    # Teacher stream: standard attention
    self.teacher_attn = _modules.Attention(
        num_heads=self.num_heads,
        num_kv_heads=self.num_kv_heads,
        features=self.features,
        head_dim=self.head_dim,
        attn_type=self.attn_type,
        query_pre_attn_scalar=self.query_pre_attn_scalar,
        rope_base_frequency=self.rope_base_frequency,
        rope_scale_factor=self.rope_scale_factor,
        attn_logits_soft_cap=self.attn_logits_soft_cap,
        sliding_window_size=self.sliding_window_size,
        use_qk_norm=self.use_qk_norm,
        name='teacher',
    )

    # Student stream: identical structure, different masking
    self.student_attn = _modules.Attention(
        num_heads=self.num_heads,
        num_kv_heads=self.num_kv_heads,
        features=self.features,
        head_dim=self.head_dim,
        attn_type=self.attn_type,
        query_pre_attn_scalar=self.query_pre_attn_scalar,
        rope_base_frequency=self.rope_base_frequency,
        rope_scale_factor=self.rope_scale_factor,
        attn_logits_soft_cap=self.attn_logits_soft_cap,
        sliding_window_size=self.sliding_window_size,
        use_qk_norm=self.use_qk_norm,
        name='student',
    )

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: _modules.LayerCache | None,
      attn_mask: jax.Array,
      *,
      deterministic: bool = False,
  ) -> tuple[_modules.LayerCache | None, jax.Array, jax.Array, jax.Array]:
    """Apply split-brain attention.

    Args:
      x: Input sequence [B, L, D].
      segment_pos: Absolute positions [B, L].
      cache: KV cache or None.
      attn_mask: Attention mask [B, L, cache_len].
      deterministic: If True, disable random masking (inference mode).

    Returns:
      Tuple of:
        - Updated cache (or None)
        - Teacher output [B, L, D]
        - Student output [B, L, D]
        - Fused output [B, L, D] (not used here, computed in block)
    """
    # Teacher stream: standard causal attention
    teacher_cache, teacher_out = self.teacher_attn(
        x, segment_pos, cache, attn_mask
    )

    # Student stream: causal + random masking
    if deterministic:
      # Inference: no random masking, Student sees same input as Teacher
      student_mask = attn_mask
    else:
      # Training: add random token masking
      student_mask = self._create_student_mask(attn_mask)

    # Note: Student uses same cache as Teacher for simplicity
    # In a full implementation, might want separate caches
    _, student_out = self.student_attn(
        x, segment_pos, cache, student_mask
    )

    return teacher_cache, teacher_out, student_out

  def _create_student_mask(self, attn_mask: jax.Array) -> jax.Array:
    """Create Student attention mask with random token masking.

    Args:
      attn_mask: Base causal mask [B, L, cache_len].

    Returns:
      Student mask with additional random masking [B, L, cache_len].
    """
    # Create column-wise mask (dropping specific tokens for ALL queries)
    # Shape: [B, 1, cache_len] or [B, 1, L]
    B, L, K = attn_mask.shape

    # Generate random mask for keys (1 = keep, 0 = drop)
    key_keep_mask = jax.random.bernoulli(
        self.make_rng('dropout'),
        p=1.0 - self.mask_ratio,
        shape=(B, 1, K),
    )

    # Always keep the current token (diagonal) visible to itself
    # to avoid complete collapse if all history is masked
    # Create diagonal mask [1, L, K]
    eye = jnp.eye(L, K, k=0, dtype=jnp.bool_)
    eye = jnp.expand_dims(eye, axis=0) # [1, L, K]

    # Broadcast key mask to [B, L, K]
    random_mask = jnp.broadcast_to(key_keep_mask, (B, L, K))

    # Ensure diagonal is always true (OR logic)
    # If a token is dropped, it's dropped from history, but t attends to t
    random_mask = random_mask | eye

    # Combine with causal mask (AND logic)
    # student_mask[b, i, j] is True if:
    # 1. causal_mask[b, i, j] is True (j <= i)
    # 2. AND (token j is kept OR i == j)
    student_mask = attn_mask & random_mask

    return student_mask


class SplitBrainBlock(nn.Module):
  """Transformer block with Split-Brain attention.

  Replaces standard attention with dual Teacher/Student streams
  and gated fusion.
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
  split_brain_config: SplitBrainConfig
  rope_base_frequency: int = _modules.DEFAULT_ROPE_BASE_FREQUENCY
  rope_scale_factor: float = _modules.DEFAULT_ROPE_SCALE_FACTOR
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  use_qk_norm: bool = False

  def setup(self):
    self.pre_attention_norm = _layers.RMSNorm()

    self.split_attn = SplitBrainAttention(
        num_heads=self.num_heads,
        num_kv_heads=self.num_kv_heads,
        features=self.embed_dim,
        head_dim=self.head_dim,
        attn_type=self.attn_type,
        query_pre_attn_scalar=self.query_pre_attn_scalar,
        mask_ratio=self.split_brain_config.mask_ratio,
        rope_base_frequency=self.rope_base_frequency,
        rope_scale_factor=self.rope_scale_factor,
        attn_logits_soft_cap=self.attn_logits_soft_cap,
        sliding_window_size=self.sliding_window_size,
        use_qk_norm=self.use_qk_norm,
    )

    self.gated_fusion = GatedFusion(
        features=self.embed_dim,
        gate_init_bias=self.split_brain_config.gate_init_bias,
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
      *,
      deterministic: bool = False,
  ) -> tuple[_modules.LayerCache | None, jax.Array, jax.Array]:
    """Apply Split-Brain block.

    Args:
      x: Input sequence [B, L, D].
      segment_pos: Absolute positions [B, L].
      cache: KV cache or None.
      attn_mask: Attention mask [B, L, cache_len].
      deterministic: If True, disable random masking.

    Returns:
      Tuple of:
        - Updated cache
        - Output sequence [B, L, D]
        - Auxiliary prophet loss (scalar)
    """
    inputs_normalized = self.pre_attention_norm(x)

    # Split-brain attention
    cache, teacher_out, student_out = self.split_attn(
        inputs_normalized,
        segment_pos,
        cache,
        attn_mask,
        deterministic=deterministic,
    )

    # Compute auxiliary prophet loss: Student(t) predicts Teacher(t+1)
    aux_loss = self._compute_prophet_loss(teacher_out, student_out)

    # Gated fusion of Teacher and Student outputs
    attn_output = self.gated_fusion(teacher_out, student_out)

    if self.post_attention_norm is not None:
      attn_output = self.post_attention_norm(attn_output)

    attn_output += x

    outputs = self.pre_ffw_norm(attn_output)
    outputs = self.mlp(outputs)

    if self.post_ffw_norm is not None:
      outputs = self.post_ffw_norm(outputs)

    outputs += attn_output

    return cache, outputs, aux_loss

  def _compute_prophet_loss(
      self,
      teacher_out: jax.Array,
      student_out: jax.Array,
  ) -> jax.Array:
    """Compute MSE loss between Student(t) and Teacher(t+1).

    Args:
      teacher_out: Teacher output [B, L, D].
      student_out: Student output [B, L, D].

    Returns:
      Scalar MSE loss.
    """
    config = self.split_brain_config

    # Shift: Student(t) predicts Teacher(t + target_shift)
    # Default shift=1: Student predicts next Teacher state
    shift = config.target_shift
    teacher_target = teacher_out[..., shift:, :]
    student_pred = student_out[..., :-shift, :]

    # Stop gradient on Teacher target to prevent collapse
    if config.stop_gradient:
      teacher_target = jax.lax.stop_gradient(teacher_target)

    # MSE loss - reduce over length and features, keep batch dim
    loss = jnp.mean((student_pred - teacher_target) ** 2, axis=(1, 2))

    return loss
