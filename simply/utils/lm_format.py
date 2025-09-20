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
  bos_id: int | None = None
  pad_id: int | None = None
  extra_eos_tokens: tuple[str, ...] = ()

  @abc.abstractmethod
  def format(self, messages: Sequence[Mapping[str, Any]]) -> str:
    """Formats the messages into string."""


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
  """LM format for Qwen R1 distill."""
  user_marker: str = '<｜User｜>'
  assistant_marker: str = '<｜Assistant｜>'
  # Note that `end_of_message_marker` is only used at the end of assistant turn.
  end_of_message_marker: str = '<｜end▁of▁sentence｜>'
  extra_eos_tokens: tuple[str, ...] = (
      '<｜User｜>', '<｜Assistant｜>', '<｜end▁of▁sentence｜>')

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


@LMFormatRegistry.register
@dataclasses.dataclass
class GeminiChat(GemmaV2Chat):
  """LM format for Gemini."""

  system_marker: str = '<ctrl99>system\n'
  user_marker: str = '<ctrl99>user\n'
  assistant_marker: str = '<ctrl99>model\n'
  end_of_message_marker: str = '<ctrl100>\n'
  extra_eos_tokens: tuple[str, ...] = ('<ctrl100>',)
