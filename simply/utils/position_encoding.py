# Copyright 2024 The Simply Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Position encoding configurations."""

import dataclasses
from typing import ClassVar

import jax.numpy as jnp

from simply.utils import registry


class PositionEncodingRegistry(registry.RootRegistry):
  """Registry for position encoding configurations."""
  namespace: ClassVar[str] = 'position_encoding'


@PositionEncodingRegistry.register
@dataclasses.dataclass(frozen=True)
class PositionEncodingConfig:
  """Base class for position encoding configurations."""

  def apply(self, embedding_mat, segment_positions=None):
    """Apply position encoding to embedding matrix.

    Args:
      embedding_mat: Input tensor of shape 
        [batch, seq_len, num_heads, head_dim].
      segment_positions: Optional position indices of shape [batch, seq_len].

    Returns:
      Tensor with position encoding applied.
    """
    raise NotImplementedError(
        f'{self.__class__.__name__}.apply() not implemented'
    )


@PositionEncodingRegistry.register
@dataclasses.dataclass(frozen=True)
class RoPE(PositionEncodingConfig):
  """Standard Rotary Position Embedding configuration.

  Attributes:
    min_timescale: Minimum timescale for frequency computation.
    max_timescale: Maximum timescale for frequency computation.
    scale_factor: Scale factor for position indices (for context length
      extension).
  """
  min_timescale: int = 1
  max_timescale: int = 10_000
  scale_factor: float = 1.0

  def apply(self, embedding_mat, segment_positions=None):
    """Apply rotary positional embedding.

    Args:
      embedding_mat: Input tensor of shape 
        [batch, seq_len, num_heads, head_dim].
      segment_positions: Optional position indices of shape [batch, seq_len].

    Returns:
      Tensor with rotary positional embedding applied.
    """
    embedding_dims = embedding_mat.shape[-1]
    half_embedding_dim = embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / embedding_dims
    timescale = self.min_timescale * (
        self.max_timescale / self.min_timescale
    ) ** fraction
    query_segment_pos = segment_positions
    if query_segment_pos is None:
      seq_length = embedding_mat.shape[1]
      query_segment_pos = jnp.arange(
          seq_length, dtype=jnp.float32
      )[jnp.newaxis, :]
    else:
      query_segment_pos = jnp.asarray(query_segment_pos, dtype=jnp.float32)
    query_segment_pos = query_segment_pos[:, :, jnp.newaxis, jnp.newaxis]
    timescale = timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = query_segment_pos / timescale / self.scale_factor
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    # Convert to float32.
    embedding_dtype = embedding_mat.dtype
    embedding_mat = jnp.asarray(embedding_mat, jnp.float32)
    first_half, second_half = jnp.split(embedding_mat, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    embedding_mat = jnp.concatenate([first_part, second_part], axis=-1)
    # Convert back to original dtype.
    embedding_mat = jnp.asarray(embedding_mat, embedding_dtype)
    return embedding_mat
