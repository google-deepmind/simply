"""Sampling helpers for Simply."""

import dataclasses
import sys

import numpy as np


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
  max_seq_len: int = sys.maxsize
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
        self.max_seq_len - 1, max_input_length + self.max_decode_steps - 1
    )
    chunk_size = self.intermediate_decode_steps
    if chunk_size is None:
      chunk_size = self.max_decode_steps
    return DecodingSchedule(
        prefill_size=prefill_size,
        begin_position=begin_position,
        end_position=end_position_exclusive,
        chunk_size=chunk_size,
    )
