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

"""Split-Brain Memory model variants.

Pre-configured Gemma models with Split-Brain memory heads.
These models use the same architecture as standard Gemma but extend the
Transformer class to support memory head training with auxiliary loss.
"""

from __future__ import annotations

from gemma.gm.ckpts import _paths
from gemma.gm.nn import _config
from gemma.gm.nn import _gemma
from gemma.gm.nn import _modules
from gemma.gm.nn._memory_attention import MemoryConfig
from gemma.gm.nn._memory_transformer import MemoryTransformer
from gemma.gm.nn._transformer import ModelInfo


class Gemma2_2B_Memory(MemoryTransformer):  # pylint: disable=invalid-name
  """Gemma2 2B with Split-Brain memory heads.

  Uses the same config as Gemma2_2B but with SplitBrainBlock for middle layers.
  Default: 50% of heads as memory heads, middle 50% of layers (layers 6-19).
  """

  config: _config.TransformerConfig = _gemma.Gemma2_2B.config

  # Default memory config for 2B model (8 heads total, so 4 memory heads)
  memory_config: MemoryConfig = MemoryConfig(
      memory_head_ratio=0.5,
      memory_loss_weight=0.1,
      target_layers=None,  # Will use middle 50% (layers 6-19 for 26 layers)
  )

  INFO = ModelInfo(
      tokenizer_version=2,
      default_ckpt=_paths.CheckpointPath.GEMMA2_2B_IT,
  )


class Gemma2_9B_Memory(MemoryTransformer):  # pylint: disable=invalid-name
  """Gemma2 9B with Split-Brain memory heads.

  Uses the same config as Gemma2_9B but with SplitBrainBlock for middle layers.
  Default: 50% of heads as memory heads, middle 50% of layers (layers 10-31).
  """

  config: _config.TransformerConfig = _gemma.Gemma2_9B.config

  memory_config: MemoryConfig = MemoryConfig(
      memory_head_ratio=0.5,
      memory_loss_weight=0.1,
      target_layers=None,  # Will use middle 50% (layers 10-31 for 42 layers)
  )

  INFO = ModelInfo(
      tokenizer_version=2,
      default_ckpt=_paths.CheckpointPath.GEMMA2_9B_IT,
  )


class Gemma3_1B_Memory(MemoryTransformer):  # pylint: disable=invalid-name
  """Gemma3 1B with Split-Brain memory heads.

  Uses the same config as Gemma3_1B but with SplitBrainBlock for middle layers.
  Default: 50% of heads as memory heads, middle 50% of layers (layers 6-19).
  """

  config: _config.TransformerConfig = _gemma.Gemma3_1B.config

  memory_config: MemoryConfig = MemoryConfig(
      memory_head_ratio=0.5,
      memory_loss_weight=0.1,
      target_layers=None,  # Will use middle 50% (layers 6-19 for 26 layers)
  )

  INFO = ModelInfo(
      tokenizer_version=3,
  )


class Gemma3_4B_Memory(MemoryTransformer):  # pylint: disable=invalid-name
  """Gemma3 4B with Split-Brain memory heads.

  Uses the same config as Gemma3_4B but with SplitBrainBlock for middle layers.
  Default: 50% of heads as memory heads, middle 50% of layers (layers 8-25).
  """

  config: _config.TransformerConfig = _gemma.Gemma3_4B.config

  memory_config: MemoryConfig = MemoryConfig(
      memory_head_ratio=0.5,
      memory_loss_weight=0.1,
      target_layers=None,  # Will use middle 50% (layers 8-25 for 34 layers)
  )

  INFO = ModelInfo(
      tokenizer_version=3,
  )
