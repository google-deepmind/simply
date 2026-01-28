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

"""Implementation of vision encoder using Simply modules."""

import dataclasses

import einops
import jax
import jax.numpy as jnp
from simply import config_lib
from simply import model_lib
from simply.utils import common
from simply.utils import initializer
from simply.utils import module


PyTree = common.PyTree
Array = common.Array
PRNGKey = jax.typing.ArrayLike
DTypeLike = jax.typing.DTypeLike
AnnotatedArray = common.AnnotatedArray


@module.ModuleRegistry.register
@dataclasses.dataclass(frozen=True)
class PatchEncoder(module.SimplyModule):
  """Patch encoding layer for vision models."""

  image_shape: tuple[int, int, int]  # hwc
  patch_size: tuple[int, int]
  encoding_dim: int
  weight_init: initializer.Initializer = initializer.XavierUniformInit()
  weight_dtype: DTypeLike = 'float32'
  activation_dtype: DTypeLike = 'float32'

  @property
  def output_grid_shape(self):
    ih, iw, _ = self.image_shape
    ph, pw = self.patch_size
    assert ih % ph == 0
    assert iw % pw == 0
    return (ih // ph), (iw // pw)

  @property
  def seq_len(self):
    oh, ow = self.output_grid_shape
    return oh * ow

  def init(self, prng_key: PRNGKey) -> PyTree:
    kernel_key, pos_key = jax.random.split(prng_key)
    params = {}
    kernel_shape = self.patch_size + (self.image_shape[-1], self.encoding_dim)
    params['kernel'] = AnnotatedArray.create(
        self.weight_init(
            kernel_key,
            shape=kernel_shape,
            dim_annotation='iiio',
            dtype=self.weight_dtype,
        ),
        dim_annotation='iiio',
    )
    params['bias'] = AnnotatedArray.create(
        jnp.zeros(
            shape=[self.encoding_dim],
            dtype=self.weight_dtype,
        )
    )
    params['pos_embedding'] = AnnotatedArray.create(
        self.weight_init(
            pos_key,
            shape=[1, self.seq_len, self.encoding_dim],
            dim_annotation='.io',
            dtype=self.weight_dtype,
        ),
        dim_annotation='.io',
    )
    return params

  def _normalize_image(self, image: Array) -> Array:
    image = (image - 127.5) / 127.5
    return jnp.clip(image, -1, 1)

  def apply(self, params: PyTree, image: Array) -> Array:
    params = common.get_raw_arrays(params)
    kernel = common.convert_or_dequantize(
        params['kernel'], dtype=self.activation_dtype
    )
    bias = common.convert_or_dequantize(
        params['bias'], dtype=self.activation_dtype
    )
    pos_embedding = common.convert_or_dequantize(
        params['pos_embedding'], dtype=self.activation_dtype
    )

    image = self._normalize_image(image)
    image = image.astype(self.activation_dtype)
    x = jax.lax.conv_general_dilated(
        image,
        kernel,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        window_strides=self.patch_size,
        padding='VALID',
    )
    x = x + jnp.expand_dims(bias, axis=(0, 1, 2))
    x = jnp.reshape(x, [x.shape[0], self.seq_len, self.encoding_dim])
    x = x + pos_embedding
    return x


@module.ModuleRegistry.register
@dataclasses.dataclass
class VisionTransformer(module.SimplyModule):
  """Simply implementation of vision encoder.

  The current implementation matches the Gemma 3 vision encoder, but may be
  extended to support other models later. The output is a sequence of soft
  embeddings which can be placed alongside text token embeddings in the main
  Gemma 3 transformer.
  """

  image_shape: tuple[int, int, int] = (896, 896, 3)
  patch_size: tuple[int, int] = (14, 14)
  width: int = 1152
  depth: int = 27
  mlp_dim: int | None = 4304
  num_heads: int = 16
  output_patch_dims: tuple[int, int] = (16, 16)
  output_embedding_dim: int = 2560
  sharding_config: model_lib.SimplyConfig | None = None

  def setup(self):
    self.patch_encoder = PatchEncoder(
        image_shape=self.image_shape,
        patch_size=self.patch_size,
        encoding_dim=self.width,
    )
    sharding_config = self.sharding_config or config_lib.gspmd_sharding()

    self.transformer_blocks = []
    for _ in range(self.depth):
      block = model_lib.TransformerBlock(
          model_dim=self.width,
          n_heads=self.num_heads,
          per_head_dim=self.width // self.num_heads,
          expand_factor=4,
          # Overrides expand_factor if present
          ffn_expand_dim=self.mlp_dim,
          qkv_use_bias=True,
          o_use_bias=True,
          attn_soft_cap=-1.0,  # disable soft cap
          norm_scale_plus_one=False,
          activation_dtype='float32',
          # Sharding related.
          sharding_config=sharding_config,
      )
      self.transformer_blocks.append(block)

    self.pre_downsample_norm = model_lib.LayerNorm(
        dim=self.width,
        scale_plus_one=False,
        # NOTE: flax.nn.LayerNorm uses float32.
        # https://github.com/google-deepmind/gemma/blame/1682669a4d568496f7a28b610a932125a48c0eb7/gemma/multimodal/vision_utils.py#L193
        activation_dtype='float32',
    )
    self.post_downsample_norm = model_lib.LayerNorm(
        dim=self.width,
        use_bias=False,
        scale_plus_one=True,
        # NOTE: flax.nn.RMSNorm uses float32.
        # https://github.com/google-deepmind/gemma/blame/1682669a4d568496f7a28b610a932125a48c0eb7/gemma/gm/nn/_modules.py#L95
        activation_dtype='float32',
    )
    self.final_projection = model_lib.EinsumLinear(
        eqn='df,...d->...f',
        weight_shape=[self.width, self.output_embedding_dim],
        activation_dtype='float32',
    )

  def init(self, prng_key: PRNGKey) -> PyTree:
    params = {}
    prng_key, patch_key = jax.random.split(prng_key)

    params['patch_encoder'] = self.patch_encoder.init(patch_key)
    for i, block in enumerate(self.transformer_blocks):
      prng_key, block_key = jax.random.split(prng_key)
      params[f'encoderblock_{i}'] = block.init(block_key)

    pre_ln_key, post_ln_key, final_proj_key = jax.random.split(prng_key, num=3)
    params['pre_downsample_ln'] = self.pre_downsample_norm.init(pre_ln_key)
    params['post_downsample_ln'] = self.post_downsample_norm.init(post_ln_key)
    params['final_projection'] = self.final_projection.init(final_proj_key)
    return params

  def downsample(
      self,
      patch_embeddings: Array,
      input_patch_dims: tuple[int, int],
      output_patch_dims: tuple[int, int],
  ) -> Array:
    ih, iw = input_patch_dims
    oh, ow = output_patch_dims

    unused_batch_size, seq_len = patch_embeddings.shape[:2]
    assert seq_len == ih * iw

    assert ih % oh == 0
    fh = ih // oh
    assert iw % ow == 0
    fw = iw // ow

    x = einops.rearrange(
        patch_embeddings,
        'b (oh fh ow fw) d -> b oh fh ow fw d',
        oh=oh,
        fh=fh,
        ow=ow,
        fw=fw,
    )
    x = jnp.mean(x, axis=[2, 4])
    x = einops.rearrange(x, 'b oh ow d -> b (oh ow) d')
    return x

  def apply(self, params: PyTree, image: Array) -> Array:
    assert self.width % self.num_heads == 0

    x = self.patch_encoder.apply(params['patch_encoder'], image)
    n, seq_len = x.shape[:2]

    # Set all positions to 0 to effectively turn off rotary embeddings. For
    # vision we rely on learned position embeddings.
    segment_positions = jnp.zeros((n, seq_len))
    segment_ids = jnp.ones_like(segment_positions)

    for i, block in enumerate(self.transformer_blocks):
      x, _ = block.apply(
          params[f'encoderblock_{i}'],
          x,
          segment_ids=segment_ids,
          segment_positions=segment_positions,
      )

    x = self.pre_downsample_norm.apply(params['pre_downsample_ln'], x)
    x = self.downsample(
        x,
        input_patch_dims=self.patch_encoder.output_grid_shape,
        output_patch_dims=self.output_patch_dims,
    )

    # "Embedder" part of Gemma 3 vision encoding that converts the
    # Siglip dimension to the internal transformer dimension.
    x = self.post_downsample_norm.apply(params['post_downsample_ln'], x)
    x = self.final_projection.apply(params['final_projection'], x)
    return x
