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
"""LM format for different models."""

import abc
from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, ClassVar
from simply.utils import registry


class LMFormatRegistry(registry.RootRegistry):
  """Evaluation registry."""
  namespace: ClassVar[str] = 'lm_format'


@dataclasses.dataclass
class LMFormat(abc.ABC):
  """Base class for Language Model formatting.

  This class defines the interface for formatting a sequence of messages
  into a single string or token sequence.
  """
  bos_id: int | None = None
  pad_id: int | None = None
  extra_eos_tokens: tuple[str, ...] = ()

  @abc.abstractmethod
  def format(self, messages: Sequence[Mapping[str, Any]]) -> str:
    """Formats the messages into string string (for inference)."""

  def format_tokens(
      self,
      messages: Sequence[Mapping[str, Any]],
      tokenizer,
      trainable_roles: tuple[str, ...] | None = None,
  ) -> tuple[list[int], list[float]]:
    """Formats and tokenizes conversation with per-token loss mask.

    This method processes messages the same way as format() but returns tokens
    and loss masks instead of a string. Subclasses should override this for
    custom tokenization behavior.

    Args:
      messages: List of message dicts with 'role' and 'content' keys.
      tokenizer: Tokenizer with encode() method.
      trainable_roles: Roles to include in loss. None = all roles trainable.

    Returns:
      tokens: List of token IDs
      loss_mask: List of floats (1.0 = in loss, 0.0 = not in loss)
    """
    all_tokens = []
    all_mask = []

    for msg in messages:
      role = msg['role']
      content = msg['content']
      is_trainable = trainable_roles is None or role in trainable_roles
      mask_value = 1.0 if is_trainable else 0.0

      # Role marker (not trainable).
      role_marker = getattr(self, f'{role}_marker', '')
      if role_marker:
        role_tokens = list(tokenizer.encode(role_marker))
        all_tokens.extend(role_tokens)
        all_mask.extend([0.0] * len(role_tokens))

      # Content (trainable based on role).
      content_tokens = list(tokenizer.encode(content))
      all_tokens.extend(content_tokens)
      all_mask.extend([mask_value] * len(content_tokens))

      # End marker (trainable if role is trainable, so model learns to stop).
      end_marker = getattr(self, 'end_of_message_marker', '')
      if end_marker:
        end_tokens = list(tokenizer.encode(end_marker))
        all_tokens.extend(end_tokens)
        all_mask.extend([mask_value] * len(end_tokens))

    return all_tokens, all_mask


@LMFormatRegistry.register
@dataclasses.dataclass
class Pretrain(LMFormat):
  """Pre-training model format."""

  def format(self, messages: Sequence[Mapping[str, Any]]) -> str:
    if len(messages) != 1:
      raise ValueError(
          f'Pre-training model only supports 1 message, got {len(messages)}.'
      )
    return messages[0]['content']


@LMFormatRegistry.register
@dataclasses.dataclass
class SimplyV1Chat(LMFormat):
  """LM format for Simply V1."""
  user_marker: str = '<reserved_1>'
  assistant_marker: str = '<reserved_2>'
  system_marker: str = '<reserved_3>'
  end_of_message_marker: str = '<reserved_4>'
  extra_eos_tokens: tuple[str, ...] = (
      '<reserved_1>', '<reserved_2>', '<reserved_3>', '<reserved_4>')
  # Training used 0 (instead of vocab.bos_id) as the bos_id and also
  # the pad_id so we keep it consistent here.
  bos_id: int = 0
  pad_id: int = 0

  def format(self, messages: Sequence[Mapping[str, Any]]) -> str:
    output = ''
    for message in messages:
      if message['role'] == 'system':
        output += self.system_marker
      elif message['role'] == 'user':
        output += self.user_marker
      elif message['role'] == 'assistant':
        output += self.assistant_marker
      else:
        raise ValueError(f'Unknown role: {message["role"]}')
      output += message['content']
      output += self.end_of_message_marker
    output += self.assistant_marker
    return output


@LMFormatRegistry.register
@dataclasses.dataclass
class GemmaV2Chat(LMFormat):
  """LM format for Gemma V2."""
  system_marker: str = '<start_of_turn>system\n'
  user_marker: str = '<start_of_turn>user\n'
  assistant_marker: str = '<start_of_turn>model\n'
  end_of_message_marker: str = '<end_of_turn>\n'
  extra_eos_tokens: tuple[str, ...] = ('<start_of_turn>', '<end_of_turn>')

  def format(self, messages: Sequence[Mapping[str, Any]]) -> str:
    output = ''
    for message in messages:
      if message['role'] == 'user':
        output += self.user_marker
      elif message['role'] == 'assistant':
        output += self.assistant_marker
      elif message['role'] == 'system':
        output += self.system_marker
      else:
        raise ValueError(f'Unknown role: {message["role"]}')
      output += message['content']
      output += self.end_of_message_marker
    output += self.assistant_marker
    return output


@LMFormatRegistry.register
@dataclasses.dataclass
class DeepSeekQwenR1DistillChat(LMFormat):
  """LM format for Qwen R1 distill.

  Note: This format only supports user/assistant roles (no system), and only
  adds end_of_message_marker after assistant turns.
  """
  user_marker: str = '<｜User｜>'
  assistant_marker: str = '<｜Assistant｜>'
  # Note that `end_of_message_marker` is only used at the end of assistant turn.
  end_of_message_marker: str = '<｜end▁of▁sentence｜>'
  extra_eos_tokens: tuple[str, ...] = (
      '<｜User｜>', '<｜Assistant｜>', '<｜end▁of▁sentence｜>')

  def format_tokens(
      self,
      messages: Sequence[Mapping[str, Any]],
      tokenizer,
      trainable_roles: tuple[str, ...] | None = None,
  ) -> tuple[list[int], list[float]]:
    """Formats and tokenizes with DeepSeek-specific handling.

    Only user/assistant roles are supported. End marker only after assistant.

    Args:
      messages: List of message dicts with 'role' and 'content' keys.
      tokenizer: Tokenizer with encode() method.
      trainable_roles: Roles to include in loss. None = all roles trainable.

    Returns:
      tokens: List of token IDs
      loss_mask: List of floats (1.0 = in loss, 0.0 = not in loss)
    """
    all_tokens = []
    all_mask = []

    for msg in messages:
      role = msg['role']
      content = msg['content']

      # Only user/assistant supported.
      if role == 'user':
        role_marker = self.user_marker
      elif role == 'assistant':
        role_marker = self.assistant_marker
      else:
        raise ValueError(
            f'DeepSeekQwenR1DistillChat does not support role: {role}'
        )

      is_trainable = trainable_roles is None or role in trainable_roles
      mask_value = 1.0 if is_trainable else 0.0

      # Role marker (not trainable).
      role_tokens = list(tokenizer.encode(role_marker))
      all_tokens.extend(role_tokens)
      all_mask.extend([0.0] * len(role_tokens))

      # Content (trainable based on role).
      content_tokens = list(tokenizer.encode(content))
      all_tokens.extend(content_tokens)
      all_mask.extend([mask_value] * len(content_tokens))

      # End marker only after assistant (trainable if assistant is trainable).
      if role == 'assistant':
        end_tokens = list(tokenizer.encode(self.end_of_message_marker))
        all_tokens.extend(end_tokens)
        all_mask.extend([mask_value] * len(end_tokens))

    return all_tokens, all_mask

  def format(self, messages: Sequence[Mapping[str, Any]]) -> str:
    output = ''
    for message in messages:
      if message['role'] == 'user':
        output += self.user_marker
      elif message['role'] == 'assistant':
        output += self.assistant_marker
      else:
        raise ValueError(f'Unknown role: {message["role"]}')
      output += message['content']
      if message['role'] == 'assistant':
        output += self.end_of_message_marker
    output += self.assistant_marker + '<think>\n'
    return output


@LMFormatRegistry.register
@dataclasses.dataclass
class QwenV2Chat(LMFormat):
  """LM format for Qwen V2."""
  user_marker: str = '<|im_start|>user\n'
  assistant_marker: str = '<|im_start|>assistant\n'
  system_marker: str = '<|im_start|>system\n'
  end_of_message_marker: str = '<|im_end|>\n'
  extra_eos_tokens: tuple[str, ...] = ('<|im_start|>', '<|im_end|>')
  add_think_marker: bool = False

  def format(self, messages: Sequence[Mapping[str, Any]]) -> str:
    output = ''
    for message in messages:
      if message['role'] == 'system':
        output += self.system_marker
      elif message['role'] == 'user':
        output += self.user_marker
      elif message['role'] == 'assistant':
        output += self.assistant_marker
      else:
        raise ValueError(f'Unknown role: {message["role"]}')
      output += message['content']
      output += self.end_of_message_marker
    output += self.assistant_marker
    if self.add_think_marker:
      output += '<think>\n'
    return output


@LMFormatRegistry.register
@dataclasses.dataclass
class QwQChat(QwenV2Chat):
  add_think_marker: bool = True
