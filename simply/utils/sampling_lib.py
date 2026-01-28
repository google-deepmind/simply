"""Sampling helpers for Simply.

The life of a sample is summarized below:

1) Initial raw input is a sequence of `Chunk`. Similar to Gemini APIs, these
   usually will simply hold text strings but can also hold arbitrary other input
   data (e.g. images).

2) `Chunk`s are processed into `ProcessedInput`, which is a purely array-based
   representation of the input. For example, converting text into tokens will
   happen here. The processing logic is specified by an implementation of
   `InputProcessorInterface`, which can be customized.

3) Multiple `ProcessedInput` are formed into `ProcessedInputBatch`, which may
   involve padding/truncation.

4) The `ProcessedInputBatch` is fed into the Jax-based sampling loop, producing
   `SamplingOutput`s. TODO: Consider moving sampling loop code from
   model_lib.py to this file.
"""

import dataclasses
import enum
import functools
from typing import ClassVar, Mapping, Protocol, Sequence

import einops
import jax
import jax.numpy as jnp
import numpy as np
from simply.utils import common
from simply.utils import distributions
from simply.utils import registry


class SamplingRegistry(registry.RootRegistry):
  """Registry for Sampling related data structures."""

  namespace: ClassVar[str] = 'Sampling'


@dataclasses.dataclass(frozen=True)
class DecodingSchedule:
  """Encapsulates the indices used for decoding in chunks.

  Decoding at position i means predicting the token in position i+1. We will
  decode intervals at a time of the form

  [prefill_size + k * chunk_size, prefill_size + (k+1) * chunk_size),

  truncated at begin_position (inclusive) and end_position (exclusive).

  Attributes:
    prefill_size: The number of tokens to prefill.
    begin_position: The minimum position to decode (inclusive).
    end_position: The maximum position to decode (exclusive).
    chunk_size: The number of tokens to decode in each chunk.
  """

  prefill_size: int
  begin_position: int
  end_position: int
  chunk_size: int

  def get_next_length(self, cur_position: int) -> int:
    if cur_position < self.begin_position:
      return self.begin_position
    step_multiple = ((cur_position - self.prefill_size) // self.chunk_size) + 1
    pos = self.prefill_size + self.chunk_size * step_multiple
    return min(pos, self.end_position)


@dataclasses.dataclass(frozen=True)
class SamplingParams:
  """Sampling parameters."""

  temperature: float = 1.0
  top_k: int = -1
  top_p: float = 1.0
  max_decode_steps: int = 256
  # `max_seq_len` is bos/eos counted, required > 0.
  # When both `max_seq_len` and `max_decode_steps` are set, we will stop
  # decoding at `min(input_len + max_decode_steps, max_seq_len)`. So when you
  # intend to leverage `max_seq_len` behavior, you usually need to make sure
  # `max_decode_steps` is large enough.
  # The typical use case to set `max_seq_len` is to sample sequences for
  # training, because training requires a fixed length.
  max_seq_len: int = np.iinfo(np.int32).max // 2
  # Maximum length of input to accept (bos counted). Longer inputs will be
  # truncated to the trailing `max_input_len` tokens.
  max_input_len: int | None = None
  num_samples: int = 1

  min_prefill_size: int = 256
  # Length to prefill, if specified. Otherwise, a prefill size will be inferred
  # from other settings.
  prefill_size: int | None = None

  intermediate_decode_steps: int | None = None  # Recommended to set.
  sort_by: str | None = None

  def get_decoding_schedule(
      self, min_input_length: int, max_input_length: int
  ) -> DecodingSchedule:
    """Creates DecodingSchedule based on the sampling params."""
    prefill_size = self.prefill_size
    if prefill_size is None:
      prefill_size = max(
          int(np.exp2(np.ceil(np.log2(min_input_length)))),
          self.min_prefill_size,
      )

    # If prefill_size >= min_input_length, we start at min_input_length - 1 so
    # that we can decode the first output token at index min_input_length.
    #
    # If prefill_size < min_input_length, we start at prefill_size, which is the
    # index of the first non-prefilled decode state.
    begin_position = min(prefill_size, min_input_length - 1)
    # For sequences of length up to L, the largest decoding position we need is
    # L - 2, which decodes for index L - 1.
    end_position_exclusive = min(
        self.max_seq_len - 1,
        # Ensure int to avoid overflow.
        int(max_input_length) + self.max_decode_steps - 1,
    )
    chunk_size = self.intermediate_decode_steps
    if chunk_size is None or chunk_size <= 0:
      chunk_size = self.max_decode_steps
    return DecodingSchedule(
        prefill_size=prefill_size,
        begin_position=begin_position,
        end_position=end_position_exclusive,
        chunk_size=chunk_size,
    )


@SamplingRegistry.register
@dataclasses.dataclass(frozen=True)
class ProcessedInput:
  tokens: Sequence[int]
  extra_inputs: Mapping[str, common.PyTree] | None = None


@SamplingRegistry.register
@dataclasses.dataclass(frozen=True)
class ProcessedInputBatch:
  """Holder for all sampling input after processing.

  Processing includes tokenization, padding, etc. The contents here are ready to
  be fed into the model.
  """

  tokens: common.Array
  lengths: common.Array
  extra_inputs: Mapping[str, common.PyTree]

  @classmethod
  def from_unpadded_inputs(
      cls, unpadded_inputs: Sequence[ProcessedInput], pad_id: int = 0
  ) -> 'ProcessedInputBatch':
    """Creates a padded version of an unpadded batch of ProcessedInput.

    Args:
      unpadded_inputs: Sequence of ProcessedInput which can have differing
        lengths/shapes.
      pad_id: ID to use for padding tokens.

    Returns:
      ProcessedInputBatch formed by padding all inputs to the same shape.

    Note that ProcessedInput.extra_inputs fields will be padded to the maximum
    size in each dimension for the batch.
    """
    lengths = [len(x.tokens) for x in unpadded_inputs]
    max_length = max(lengths)

    tokens = []
    for processed_input in unpadded_inputs:
      pad_size = max_length - len(processed_input.tokens)
      tokens.append(list(processed_input.tokens) + [pad_id] * pad_size)

    def form_batch(*xs: common.Array):
      present_xs = [x for x in xs if x is not None]
      if not present_xs:
        return None
      max_shape = np.max(np.array([x.shape for x in present_xs]), axis=0)
      batch_array = np.zeros((len(xs), *max_shape), dtype=present_xs[0].dtype)
      for i, x in enumerate(xs):
        if x is not None:
          indices = (i,) + tuple(slice(0, d) for d in x.shape)
          batch_array[indices] = x
      return batch_array

    extra_inputs = jax.tree_util.tree_map(
        form_batch, *[x.extra_inputs for x in unpadded_inputs]
    )

    return ProcessedInputBatch(
        tokens=np.array(tokens),
        lengths=np.array(lengths),
        extra_inputs=extra_inputs,
    )

  @property
  def batch_size(self):
    return self.tokens.shape[0]

  @property
  def min_length(self):
    return min(self.lengths)

  @property
  def max_length(self):
    return max(self.lengths)

  def token_slice(self, start, end):
    return self.tokens[:, start:end]

  def pad_to(self, length, pad_id=0):
    pad_len = length - self.tokens.shape[-1]
    if pad_len <= 0:
      return self
    padded_tokens = np.pad(
        self.tokens, ((0, 0), (0, pad_len)), constant_values=pad_id
    )
    return dataclasses.replace(self, tokens=padded_tokens)

  def repeat(self, n: int) -> 'ProcessedInputBatch':
    repeat_fn = lambda x: einops.repeat(x, 'b ... -> (b n) ...', n=n)
    return ProcessedInputBatch(
        tokens=repeat_fn(self.tokens),
        lengths=repeat_fn(self.lengths),
        extra_inputs=jax.tree_util.tree_map(repeat_fn, self.extra_inputs),
    )

  def pad_batch_to(self, batch_size: int) -> 'ProcessedInputBatch':

    def pad_fn(x):
      paddings = [(0, batch_size - self.batch_size)]
      paddings += [(0, 0) for _ in range(len(x.shape) - 1)]
      return np.pad(x, paddings)

    return ProcessedInputBatch(
        tokens=pad_fn(self.tokens),
        lengths=pad_fn(self.lengths),
        extra_inputs=jax.tree_util.tree_map(pad_fn, self.extra_inputs),
    )


@SamplingRegistry.register
@dataclasses.dataclass(frozen=True)
class Chunk:

  @functools.partial(SamplingRegistry.register, name='Chunk.Type')
  class Type(enum.StrEnum):
    TEXT = 'text'
    ARRAY = 'array'

  type: 'Chunk.Type'
  role: str | None = None
  content: str | bytes | common.Array | None = None


ChunkSequence = Sequence[Chunk]
# ChunkSequence is the most general input format, but we support string as well
# for backward compatibility. We also want to make sure the text-only use case
# has an easy interface.
SamplingInput = str | ChunkSequence


def chunks_as_text(chunks: ChunkSequence):
  texts = [c.content for c in chunks if c.type == Chunk.Type.TEXT]
  return ''.join(texts)


def input_as_chunks(sampling_input: SamplingInput) -> ChunkSequence:
  if isinstance(sampling_input, str):
    return [Chunk(type=Chunk.Type.TEXT, content=sampling_input)]
  return sampling_input


def input_as_text(sampling_input: SamplingInput) -> str:
  if isinstance(sampling_input, str):
    return sampling_input
  return chunks_as_text(sampling_input)


class InputProcessorRegistry(registry.RootRegistry):
  """Input processor registry."""

  namespace: ClassVar[str] = 'input_processor'


class InputProcessorInterface(Protocol):
  """Generic interface for preprocessing input.

  This class takes in raw user input in the form of chunks and produces output
  arrays ready to be fed into Jax. This interface operates on a single example,
  with padding handled by ProcessedInputBatch.

  Note: If extra_inputs shapes are not the same across examples, be aware that
  padding will be automatically added for batching, and the resulting shapes may
  trigger multiple JIT compilations (see ProcessedInputBatch for details).
  """

  eos_ids: list[int]
  pad_id: int

  def encode(
      self, chunks: ChunkSequence, max_input_len: int | None = None
      ) -> ProcessedInput:
    ...

  def decode(self, token_ids: list[int]) -> ChunkSequence:
    ...

  def input_as_chunks(self, sampling_input: SamplingInput) -> ChunkSequence:
    # Also expose through input processor so that the user don't need to
    # also import sampling_lib to handle raw `sampling_input`.
    return input_as_chunks(sampling_input)


@InputProcessorRegistry.register
class BasicTextInputProcessor(InputProcessorInterface):
  """Basic input processor for text."""

  def __init__(
      self,
      vocab,
      bos_id_override: int | None = None,
      pad_id_override: int | None = None,
      extra_eos_ids: Sequence[int] | None = None,
      extra_eos_tokens: Sequence[str] | None = None,
  ):
    self.vocab = vocab
    self.eos_ids = [self.vocab.eos_id]
    if extra_eos_ids is not None:
      self.eos_ids.extend(extra_eos_ids)
    if extra_eos_tokens is not None:
      for token in extra_eos_tokens:
        encoded_token_ids = vocab.encode(token)
        assert len(encoded_token_ids) == 1, (
            f'Invalid eos token {token} , '
            'valid eos token must be a single token in vocab.'
        )
        self.eos_ids.append(encoded_token_ids[0])
    self.eos_ids = list(set(self.eos_ids))

    self.pad_id = (
        pad_id_override if pad_id_override is not None else vocab.pad_id
    )
    self.bos_id = (
        bos_id_override if bos_id_override is not None else vocab.bos_id
    )

  def encode(
      self, chunks: ChunkSequence,
      max_input_len: int | None = None) -> ProcessedInput:
    tokens = []
    if self.bos_id is not None:
      tokens.append(self.bos_id)
    for c in chunks:
      if c.type == Chunk.Type.TEXT:
        tokens += self.vocab.encode(c.content)
      else:
        raise ValueError(
            f'Unsupported chunk type for BasicTextInputProcessor: {c.type}')
    if max_input_len is not None:
      tokens = tokens[-max_input_len:]
    return ProcessedInput(tokens=tokens)

  def decode(self, token_ids: Sequence[int]) -> ChunkSequence:
    if token_ids and token_ids[0] == self.bos_id:
      text = self.vocab.decode(token_ids[1:])
    else:
      text = self.vocab.decode(token_ids)
    return [Chunk(type=Chunk.Type.TEXT, content=text)]


@InputProcessorRegistry.register
class EmbeddingTextInputProcessor(BasicTextInputProcessor):
  """Input processor that allows precomputed embeddings."""

  def encode(
      self, chunks: ChunkSequence,
      max_input_len: int | None = None) -> ProcessedInput:
    tokens = [self.bos_id]
    extra_inputs = None
    for c in chunks:
      if c.type == Chunk.Type.TEXT:
        tokens += self.vocab.encode(c.content)
      elif c.type == Chunk.Type.ARRAY:
        assert extra_inputs is None, 'Only one array chunk is allowed.'
        if isinstance(c.content, common.Array):
          assert (
              c.content.ndim == 2
          ), 'Embeddings must be of shape (num_tokens, dim).'
          extra_inputs = {'embeddings': c.content}
    if max_input_len is not None:
      # NOTE: This may truncate the embeddings.
      tokens = tokens[-max_input_len:]
    return ProcessedInput(tokens=tokens, extra_inputs=extra_inputs)


def create_input_processor(config, **kwargs) -> InputProcessorInterface:
  """Creates input processor from config."""
  input_processor_name = getattr(config, 'input_processor_name', None)
  if input_processor_name is None:
    return BasicTextInputProcessor(**kwargs)
  return InputProcessorRegistry.get(input_processor_name)(**kwargs)


def top_k_mask(logits: jax.Array, top_k: int) -> jax.Array:
  inner_size = logits.shape[-1]
  indices = jnp.argsort(logits, axis=-1, descending=True)
  mask = jnp.arange(inner_size) < top_k
  mask = jnp.broadcast_to(mask, indices.shape)
  _, mask = jax.lax.sort_key_val(indices, mask, dimension=-1)
  return mask


def top_p_mask(logits: jax.Array, top_p: float) -> jax.Array:
  probs = jax.nn.softmax(logits, axis=-1)
  indices = jnp.argsort(logits, axis=-1, descending=True)
  sorted_probs = jnp.take_along_axis(probs, indices, axis=-1)
  cumsum = jnp.cumulative_sum(sorted_probs, axis=-1, include_initial=True)
  mask = cumsum[..., :-1] < top_p
  _, mask = jax.lax.sort_key_val(indices, mask, dimension=-1)
  return mask


def sample_from_logits(
    prng_key: jax.Array,
    logits: jax.Array,
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
  """Samples from the last step of the logits.

  Args:
    prng_key: A PRNGKey used as the random key.
    logits: The logits from the model.
    temperature: The temperature for sampling.
    top_k: The maximum number of top tokens to sample from. If top_k == -1,
      results are sampled from the whole vocabulary.
    top_p: The cumulate probabilty of top tokens to sample from.

  Returns:
    The sampled tokens and the corresponding logprobs.
  """
  logits = jnp.astype(logits, jnp.float32)

  def greedy_fn(logits: jax.Array) -> tuple[jax.Array, jax.Array]:
    tokens = jnp.argmax(logits, axis=-1)
    logprobs = jnp.zeros(logits.shape[:-1], dtype=logits.dtype)
    return tokens, logprobs

  def simple_sample_fn(logits: jax.Array) -> tuple[jax.Array, jax.Array]:
    logits = logits / temperature
    m = distributions.Categorical(logits)
    tokens = m.sample(prng_key)
    logprobs = m.log_prob(tokens)
    return tokens, logprobs

  def masked_sample_fn(logits: jax.Array) -> tuple[jax.Array, jax.Array]:
    logits = logits / temperature

    mask = jax.lax.cond(
        jnp.logical_and(top_k > 0, top_p < 1),
        lambda: (
            top_k_mask(logits, top_k=top_k) & top_p_mask(logits, top_p=top_p)
        ),
        lambda: jax.lax.cond(
            top_k > 0,
            lambda: top_k_mask(logits, top_k=top_k),
            lambda: top_p_mask(logits, top_p=top_p),
        ),
    )
    m = distributions.MaskedCategorical(
        logits, mask=mask, neg_inf=common.neg_inf(logits.dtype)
    )
    tokens = m.sample(prng_key)
    logprobs = m.log_prob(tokens)
    return tokens, logprobs

  def sample_fn(logits: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jax.lax.cond(
        jnp.logical_or(top_k > 0, top_p < 1),
        masked_sample_fn,
        simple_sample_fn,
        logits,
    )
  return jax.lax.cond(
      jnp.logical_or(temperature == 0, top_k == 1), greedy_fn, sample_fn, logits
  )


def compute_log_likelihood(
    logits: jax.Array,
    tokens: jax.Array,
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
) -> jax.Array:
  """Computes the log likelyhood in float32."""
  logits = jnp.astype(logits, jnp.float32)

  def greedy_score_fn(logits: jax.Array, tokens: jax.Array) -> jax.Array:
    shape = tokens.shape
    dtype = logits.dtype
    target_tokens = jnp.argmax(logits, axis=-1)
    return jnp.where(
        tokens == target_tokens,
        jnp.zeros(shape, dtype=dtype),
        jnp.full(shape, common.neg_inf(dtype), dtype=dtype),
    )

  def simple_sample_score_fn(logits: jax.Array, tokens: jax.Array) -> jax.Array:
    # The shape of logits is (batch_size, seq_len, vocab_size).
    m = distributions.Categorical(logits / temperature)
    return m.log_prob(tokens)

  def masked_sample_score_fn(logits: jax.Array, tokens: jax.Array) -> jax.Array:
    # The shape of logits is (batch_size, seq_len, vocab_size).
    logits = logits / temperature

    mask = jax.lax.cond(
        jnp.logical_and(top_k > 0, top_p < 1),
        lambda: (
            top_k_mask(logits, top_k=top_k) & top_p_mask(logits, top_p=top_p)
        ),
        lambda: jax.lax.cond(
            top_k > 0,
            lambda: top_k_mask(logits, top_k=top_k),
            lambda: top_p_mask(logits, top_p=top_p),
        ),
    )

    m = distributions.MaskedCategorical(
        logits, mask=mask, neg_inf=common.neg_inf(logits.dtype)
    )
    return m.log_prob(tokens)

  def sample_score_fn(logits: jax.Array, tokens: jax.Array) -> jax.Array:
    return jax.lax.cond(
        jnp.logical_or(top_k > 0, top_p < 1),
        masked_sample_score_fn,
        simple_sample_score_fn,
        logits,
        tokens,
    )

  return jax.lax.cond(
      jnp.logical_or(temperature == 0, top_k == 1),
      greedy_score_fn,
      sample_score_fn,
      logits,
      tokens,
  )
