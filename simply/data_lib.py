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
r"""Utilities for dataset creation.
"""

import dataclasses
import functools
import json
import os
import random
from typing import Any, Callable, ClassVar, Mapping, MutableMapping, Optional, Protocol, Union

import einops
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
import seqio
from simply.utils import registry
from simply.utils import tokenization
import t5.data.preprocessors
import tensorflow as tf


################################################################################
# Type aliases.
Batch = MutableMapping[str, Union[np.ndarray, jnp.ndarray]]
Processor = Callable[[Batch], Batch]

DATASETS_DIR = os.getenv('SIMPLY_DATASETS', os.path.expanduser('~/.cache/simply/datasets/'))
VOCABS_DIR = os.getenv('SIMPLY_VOCABS', os.path.expanduser('~/.cache/simply/vocabs/'))

################################################################################
# Tokenizers / vocabularies.

OPENMIX_V1_32768_VOCAB = os.path.join(VOCABS_DIR, 'spm-32768-open_mix_v2_edu-r100-v1p1-07122024.model')
OPENMIX_V1_100864_VOCAB = os.path.join(VOCABS_DIR, 'spm-100864-open_mix_v1-reserved_100-02272024.model')
FWEDU_100864_V1_VOCAB = os.path.join(VOCABS_DIR, 'spm-100864-fwedu-r100-v1-07102024.model')
OPENMIX_V2_EDU_100864_VOCAB = os.path.join(VOCABS_DIR, 'spm-100864-open_mix_v2_edu-r100-v1-07122024.model')
OPENMIX_V2_EDU_100864_V1P1_VOCAB = os.path.join(VOCABS_DIR, 'spm-100864-open_mix_v2_edu-r100-v1p1-07122024.model')
OPENMIX_V3_100864_V1_VOCAB = os.path.join(VOCABS_DIR, 'spm-100864-openmix_v3-r100-v1-08312024.model')
OPENMIX_V3_100864_V2_VOCAB = os.path.join(VOCABS_DIR, 'spm-100864-openmix_v3-r100-v2-08312024.model')
GEMMA2_VOCAB = os.path.join(VOCABS_DIR, 'gemma2_tokenizer.model')
GEMMA3_VOCAB = os.path.join(VOCABS_DIR, 'gemma3_cleaned_262144_v2.spiece.model')

OPENMIX_V1_VOCABS = [
    ('vb100864_openmix_v1', OPENMIX_V1_100864_VOCAB),
    ('vb32768_openmix_v1', OPENMIX_V1_32768_VOCAB)]
OPENMIX_V2_VOCABS = [
    ('vb100864_v1p1_openmix_v2_edu', OPENMIX_V2_EDU_100864_V1P1_VOCAB)]
OPENMIX_V3_VOCABS = [
    ('vb100864_v2_openmix_v3', OPENMIX_V3_100864_V2_VOCAB)]
GEMMA2_VOCABS = [('vb256128_gemma2', GEMMA2_VOCAB)]
T5_CC_VOCABS = [
    ('vb32000_t5_cc',
     'gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model')]


def register_vocabs():
  vocabs = (
      OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS +
      OPENMIX_V3_VOCABS + GEMMA2_VOCABS)
  for name, vocab_path in vocabs:
    tokenization.TokenizerRegistry.register_value(
        seqio.SentencePieceVocabulary(vocab_path), name=name)

register_vocabs()

tokenization.TokenizerRegistry.register_value(
    seqio.SentencePieceVocabulary(GEMMA3_VOCAB), name='vb262144_gemma3'
)

PILE_50432_V1_VOCAB = os.path.join(VOCABS_DIR, 'spm-50432-pile-train00-02122024.model')
PILE_50432_V2_VOCAB = os.path.join(VOCABS_DIR, 'spm-50432-pile-train00+01-02122024.model')
PILE_50432_V3_VOCAB = os.path.join(VOCABS_DIR, 'spm-50432-pile-train00-spc2_24-02252024.model')
PILE_100864_V1_VOCAB = os.path.join(VOCABS_DIR, 'spm-100864-pile-train00+01-02142024.model')
PILE_256000_V1_VOCAB = os.path.join(VOCABS_DIR, 'spm-256000-pile-train00+01-02162024.model')

PILE_VOCABS = [
    ('vb50432_v3_pile', PILE_50432_V3_VOCAB),
    ('vb100864_v1_pile', PILE_100864_V1_VOCAB),
    ('vb256000_v1_pile', PILE_256000_V1_VOCAB),
]


USER_TOKEN = '<reserved_1>'
ASSISTANT_TOKEN = '<reserved_2>'
SYSTEM_TOKEN = '<reserved_3>'
END_OF_MESSAGE_TOKEN = '<reserved_4>'


################################################################################
# PT datasets.


def add_pt_task_v1(name, source, vocab, add_eos=False,
                   use_reduce_concat_split=True):
  preprocessors = [
      functools.partial(
          t5.data.preprocessors.rekey,
          key_map={
              'inputs': None,
              'targets': 'text',
          },
      ),
      seqio.preprocessors.tokenize,
      # Note that append_eos will respect the `add_eos`` field in
      # `output_features``.
      seqio.preprocessors.append_eos,
  ]
  if use_reduce_concat_split:
    preprocessors += [
        t5.data.preprocessors.reduce_concat_tokens,
        t5.data.preprocessors.split_tokens_to_targets_length,
    ]
  seqio.TaskRegistry.remove(name)
  seqio.TaskRegistry.add(
      name,
      source=source,
      preprocessors=preprocessors,
      output_features={
          'targets': seqio.Feature(
              seqio.SentencePieceVocabulary(vocab),
              add_eos=add_eos, dtype=tf.int32
              ),
          },
  )


def add_lm1b_task():
  lm1b_source = seqio.TfdsDataSource(
      tfds_name='lm1b:1.1.0',
      splits={
          'train': 'train[:90%]',
          'validation': 'train[90%:]',
          'test': 'test'})
  minilm1b_source = seqio.TfdsDataSource(
      tfds_name='lm1b:1.1.0',
      splits={
          'train': 'train[:500]',
          'validation': 'train[500:1000]',
          'test': 'test'})
  vocabs = OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS
  vocabs += [('vb32768_openmix_v1', OPENMIX_V1_32768_VOCAB)]
  for name, source in [('lm1b', lm1b_source),
                       ('minilm1b', minilm1b_source)]:
    for vocab_name, vocab in vocabs:
      task_name = f'{name}.{vocab_name}'
      add_pt_task_v1(task_name, source, vocab,
                     use_reduce_concat_split=False)

add_lm1b_task()


def add_c4_task():
  source = seqio.TfdsDataSource(tfds_name='c4:3.0.1')
  vocabs = OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS
  for vocab_name, vocab in vocabs:
    task_name = f'c4.{vocab_name}'
    add_pt_task_v1(task_name, source, vocab,
                   use_reduce_concat_split=True)
add_c4_task()


def add_imdb_reviews_task():
  """Adds imdb_reviews tasks."""
  source = seqio.TfdsDataSource(
      tfds_name='imdb_reviews:1.0.0',
      splits={
          'train': 'train[:90%]',
          'validation': 'train[90%:]',
          'test': 'test'})
  name = 'imdb_reviews'
  for vocab_name, vocab in T5_CC_VOCABS:
    task_name = f'{name}.{vocab_name}'
    add_pt_task_v1(task_name, source, vocab,
                   use_reduce_concat_split=False)

add_imdb_reviews_task()


def add_pile_tasks():
  the_pile_train = os.path.join(DATASETS_DIR, 'pile/pile_tfrecord/train.tfrecord*')
  the_pile_validation = os.path.join(DATASETS_DIR, 'pile/pile_tfrecord/val.tfrecord*')
  the_pile_test = os.path.join(DATASETS_DIR, 'pile/pile_tfrecord/test.tfrecord*')
  the_pile_source = seqio.TFExampleDataSource(
      split_to_filepattern={
          'train': the_pile_train,
          'validation': the_pile_validation,
          'test': the_pile_test},
      feature_description={
          'text': tf.io.FixedLenFeature([], dtype=tf.string),
          'source': tf.io.FixedLenFeature([], dtype=tf.string)})
  for vocab_name, vocab in PILE_VOCABS:
    task_name = f'the_pile_lm.{vocab_name}'
    add_pt_task_v1(task_name, the_pile_source, vocab)

add_pile_tasks()


# Add redpajama_1t datasets.
def add_redpajama_1t_task():
  for cat in ['arxiv', 'wikipedia', 'book', 'stackexchange']:
    path = os.path.join(DATASETS_DIR, 'redpajama_1t/tfrecord/{cat}.tfrecord*')
    source = seqio.TFExampleDataSource(
        split_to_filepattern={'train': path},
        feature_description={
            'text': tf.io.FixedLenFeature([], dtype=tf.string)})
    for vocab_name, vocab in (OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS +
                              OPENMIX_V3_VOCABS):
      task_name = f'redpajama_1t_{cat}.{vocab_name}'
      add_pt_task_v1(task_name, source, vocab)

add_redpajama_1t_task()


# Add starcoder datasets
def add_starcoder_task():
  path = os.path.join(DATASETS_DIR, 'starcoder/tfrecord/train.tfrecord*')
  source = seqio.TFExampleDataSource(
      split_to_filepattern={'train': path},
      feature_description={
          'text': tf.io.FixedLenFeature([], dtype=tf.string)})
  for vocab_name, vocab in OPENMIX_V1_VOCABS:
    task_name = f'starcoder.{vocab_name}'
    add_pt_task_v1(task_name, source, vocab)
add_starcoder_task()


# Add refinedweb datasets
def add_refinedweb_task():
  path = os.path.join(DATASETS_DIR, 'refinedweb/tfrecord/train.tfrecord*')
  source = seqio.TFExampleDataSource(
      split_to_filepattern={'train': path},
      feature_description={
          'text': tf.io.FixedLenFeature([], dtype=tf.string)})
  for vocab_name, vocab in OPENMIX_V1_VOCABS:
    task_name = f'refinedweb.{vocab_name}'
    add_pt_task_v1(task_name, source, vocab)

add_refinedweb_task()


def add_fineweb_edu_task():
  path = os.path.join(DATASETS_DIR, 'fineweb-edu/train1.tfrecord-*')
  source = seqio.TFExampleDataSource(
      split_to_filepattern={'train': path},
      feature_description={
          'text': tf.io.FixedLenFeature([], dtype=tf.string)})

  for vocab_name, vocab in ([
      ['fwedu_100864_v1', FWEDU_100864_V1_VOCAB]] +
                            OPENMIX_V1_VOCABS +
                            OPENMIX_V2_VOCABS):
    task_name = f'fineweb_edu.{vocab_name}'
    add_pt_task_v1(task_name, source, vocab)

add_fineweb_edu_task()


def add_dclm_baseline_1p0_task():
  path = os.path.join(DATASETS_DIR, 'dclm-baseline-1p0/tfrecords/*/*')
  source = seqio.TFExampleDataSource(
      split_to_filepattern={'train': path},
      feature_description={
          'text': tf.io.FixedLenFeature([], dtype=tf.string)})

  for vocab_name, vocab in (OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS +
                            OPENMIX_V3_VOCABS):
    task_name = f'dclm_baseline_1p0.{vocab_name}'
    add_pt_task_v1(task_name, source, vocab)

add_dclm_baseline_1p0_task()


def add_stack_v2_smol_task():
  repo_version_path = os.path.join(DATASETS_DIR, 'stack_v2/download/train-smol-1/train1.tfrecord*')
  file_version_path = os.path.join(DATASETS_DIR, 'stack_v2/download/train-smol-1-file/train2.tfrecord*')
  for name, path in [('stack_v2_smol_repo', repo_version_path),
                     ('stack_v2_smol_file', file_version_path)]:
    source = seqio.TFExampleDataSource(
        split_to_filepattern={'train': path},
        feature_description={
            'text': tf.io.FixedLenFeature([], dtype=tf.string)})
    for vocab_name, vocab in (OPENMIX_V2_VOCABS + OPENMIX_V3_VOCABS):
      task_name = f'{name}.{vocab_name}'
      add_pt_task_v1(task_name, source, vocab)

add_stack_v2_smol_task()


################################################################################
# SFT datasets.


def converation_preprocessor(
    dataset: tf.data.Dataset, fn: Callable[..., str]) -> tf.data.Dataset:

  @seqio.map_over_dataset
  def construct_conversation_map(
      ex: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    def py_func(json_str):
      serialized_conversation = json_str.numpy().decode('utf-8')
      return fn(serialized_conversation)
    result_tensor = tf.py_function(
        func=py_func, inp=[ex['conversation']], Tout=tf.string)
    result_tensor.set_shape([])
    return {
        'conversation': result_tensor,
    }
  return construct_conversation_map(dataset)


def add_sft_task_v1(name, source, vocab, conversation_process_fn):
  seqio.TaskRegistry.remove(name)
  seqio.TaskRegistry.add(
      name,
      source=source,
      preprocessors=[
          functools.partial(
              converation_preprocessor,
              fn=conversation_process_fn),
          functools.partial(
              t5.data.preprocessors.rekey,
              key_map={
                  'inputs': None,
                  'targets': 'conversation',
              },
          ),
          seqio.preprocessors.tokenize,
          seqio.preprocessors.append_eos,
          ],
      output_features={
          'targets': seqio.Feature(
              seqio.SentencePieceVocabulary(vocab),
              add_eos=False, dtype=tf.int32
              ),
          },
  )


def process_conversation(serialized_conversation):
  conversation = json.loads(serialized_conversation)
  text = []
  role_token_dict = {
      'user': USER_TOKEN,
      'assistant': ASSISTANT_TOKEN,
      'system': SYSTEM_TOKEN}
  for message in conversation:
    content = message['content']
    role = message['role']
    text.append(f'{role_token_dict[role]}{content}{END_OF_MESSAGE_TOKEN}')
  return ''.join(text)


def add_openhermes_2p5_task():
  train = os.path.join(DATASETS_DIR, 'openhermes-2p5/train.tfrecord')
  source = seqio.TFExampleDataSource(
      split_to_filepattern={'train': train},
      feature_description={
          'conversation': tf.io.FixedLenFeature([], dtype=tf.string),
          'metadata': tf.io.FixedLenFeature([], dtype=tf.string)})
  for vocab_name, vocab in (
      OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS + OPENMIX_V3_VOCABS):
    add_sft_task_v1(
        f'openhermes_2p5.{vocab_name}', source, vocab,
        conversation_process_fn=process_conversation)

add_openhermes_2p5_task()


def add_tulu_v2_task():
  train = os.path.join(DATASETS_DIR, 'tulu-v2-sft-mixture/train.tfrecord')
  source = seqio.TFExampleDataSource(
      split_to_filepattern={'train': train},
      feature_description={
          'conversation': tf.io.FixedLenFeature([], dtype=tf.string),
          'metadata': tf.io.FixedLenFeature([], dtype=tf.string)})
  for vocab_name, vocab in OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS:
    add_sft_task_v1(
        f'tulu_v2_sft.{vocab_name}', source, vocab,
        conversation_process_fn=process_conversation)

add_tulu_v2_task()

################################################################################
# Mixtures.


# ###############################################################################
# # Dataset utilities.


class DataSourceRegistry(registry.RootRegistry):
  """Data source registry."""
  namespace: ClassVar[str] = 'datasource'


class SimpleDataSource(Protocol):

  def __len__(self):
    ...

  def __getitem__(self, index: int):
    ...


@functools.partial(DataSourceRegistry.register, name='simply_json:gsm8k_train')
@dataclasses.dataclass(frozen=True)
class GSM8KJSONTrain(SimpleDataSource):
  """GSM8K dataset in json format."""
  path: str = os.path.join(DATASETS_DIR, 'gsm8k/gsm8k.json')
  example_start_index: int | None = None
  example_end_index: int | None = None
  split: str = 'train'

  def load(self):
    with epath.Path(self.path).open('r') as f:
      data = json.load(f)
    examples = data[self.split]
    for i, example in enumerate(examples):
      example['uid'] = f'gsm8k_{self.split}-{i}'
      example['id'] = i
    return examples[self.example_start_index:self.example_end_index]


@functools.partial(DataSourceRegistry.register, name='simply_json:gsm8k_test')
@dataclasses.dataclass(frozen=True)
class GSM8KJSONTest(GSM8KJSONTrain):
  split: str = 'test'


def register_gsm8k_json_variants():
  config = GSM8KJSONTrain()
  for num_examples in [4, 32, 128]:
    new_config = dataclasses.replace(
        config, example_start_index=0, example_end_index=num_examples)
    DataSourceRegistry.register_value(
        new_config, name=f'simply_json:gsm8k_train{num_examples}')

register_gsm8k_json_variants()


@functools.partial(
    DataSourceRegistry.register, name='simply_json:simple_qa_test'
)
@dataclasses.dataclass(frozen=True)
class SimpleQATest(SimpleDataSource):
  """Simple QA dataset in json format.

  Source: https://openai.com/index/introducing-simpleqa/
  """

  path: str = os.path.join(DATASETS_DIR, 'simple_qa/simple_qa_test_set.json')
  split: str = 'test'

  def load(self):
    with epath.Path(self.path).open('r') as f:
      data = json.load(f)
    examples = data[self.split]
    for i, example in enumerate(examples):
      example['uid'] = f'simple_qa_{self.split}-{i}'
      example['id'] = i
    return examples


@functools.partial(
    DataSourceRegistry.register, name='simply_json:simple_qa_num'
)
@dataclasses.dataclass(frozen=True)
class SimpleQATestNumberOnly(SimpleQATest):
  """Simple QA dataset with only number-only answers."""

  path: str = os.path.join(
      DATASETS_DIR, 'simple_qa/simple_qa_test_set_number_only.json')


@functools.partial(DataSourceRegistry.register, name='simply_json:mmlu_test')
@dataclasses.dataclass(frozen=True)
class MMLUJSONTest(SimpleDataSource):
  """MMLU dataset in json format."""
  path: str = os.path.join(DATASETS_DIR, 'mmlu/mmlu.json')
  example_start_index: int | None = None
  example_end_index: int | None = None
  split: str = 'test'

  def load(self):
    with epath.Path(self.path).open('r') as f:
      data = json.load(f)
    examples = data['data'][self.split]
    for i, example in enumerate(examples):
      example['uid'] = f'mmlu_{self.split}-{i}'
      example['id'] = i
    return examples[self.example_start_index:self.example_end_index]


@functools.partial(
    DataSourceRegistry.register, name='simply_json:dsr40k_train')
@dataclasses.dataclass(frozen=True)
class DeepScaleRJSONTrain(SimpleDataSource):
  """DeepScaleR dataset in json format."""
  path: str = os.path.join(DATASETS_DIR, 'deepscaler/deepscaler.json')
  example_start_index: int | None = None
  example_end_index: int | None = None

  def load(self):
    with epath.Path(self.path).open('r') as f:
      examples = json.load(f)
    new_examples = []
    for i, example in enumerate(examples):
      new_examples.append({
          'question': example['problem'],
          'short_answer': example['answer'],
          'answer': example['solution'],
          'uid': f'dsr40k_train-{i}',
          'id': i,
      })
    return new_examples[self.example_start_index:self.example_end_index]


# TODO: add a unified interface for filtering AIME examples
@functools.partial(
    DataSourceRegistry.register, name='simply_json:aime24')
@dataclasses.dataclass(frozen=True)
class AIME24JSON(SimpleDataSource):
  """AIME24 dataset in json format."""
  path: str = os.path.join(DATASETS_DIR, 'aime/aime_v2.json')
  example_start_index: int | None = None
  example_end_index: int | None = None

  def load(self):
    with epath.Path(self.path).open('r') as f:
      examples = json.load(f)
    new_examples = []
    for i, example in enumerate(examples):
      if int(example['year']) == 2024:
        # using the same keys as DeepScaleR
        new_examples.append({
            'question': example['problem'],
            'short_answer': example['answer'],
            'answer': example['solution'],
            'uid': f'aime24-{i}',
            'id': i,
        })
    return new_examples[self.example_start_index:self.example_end_index]


@functools.partial(
    DataSourceRegistry.register, name='simply_json:aime25')
@dataclasses.dataclass(frozen=True)
class AIME25JSON(SimpleDataSource):
  """AIME25 dataset in json format."""
  path: str = os.path.join(DATASETS_DIR, 'aime/aime_v2.json')
  example_start_index: int | None = None
  example_end_index: int | None = None

  def load(self):
    with epath.Path(self.path).open('r') as f:
      examples = json.load(f)
    new_examples = []
    for i, example in enumerate(examples):
      if int(example['year']) == 2025:
        # using the same keys as DeepScaleR
        new_examples.append({
            'question': example['problem'],
            'short_answer': example['answer'],
            'answer': example['solution'],
            'uid': f'aime25-{i}',
            'id': i,
        })
    return new_examples[self.example_start_index:self.example_end_index]


# TODO: check the 14B eval accuracy
@functools.partial(
    DataSourceRegistry.register, name='simply_json:math500_test')
@dataclasses.dataclass(frozen=True)
class MATH500JSONTest(SimpleDataSource):
  """MATH500 test set in json format."""
  path: str = os.path.join(DATASETS_DIR, 'math500/test.json')
  example_start_index: int | None = None
  example_end_index: int | None = None

  def load(self):
    with epath.Path(self.path).open('r') as f:
      examples = json.load(f)
    new_examples = []
    for i, example in enumerate(examples):
      # using the same keys as DeepScaleR
      new_examples.append({
          'question': example['problem'],
          'short_answer': example['answer'],
          'answer': example['solution'],
          'subject': example['subject'],
          'level': example['level'],
          'original_unique_id': example['unique_id'],
          'uid': f'math500_test-{i}',
          'id': i,
      })
    return new_examples[self.example_start_index:self.example_end_index]


# TODO: check the 14B eval accuracy
@functools.partial(
    DataSourceRegistry.register, name='simply_json:gpqa_diamond')
@dataclasses.dataclass(frozen=True)
class GPQADiamondJSON(SimpleDataSource):
  """GPQA-Diamond dataset in json format."""
  path: str = os.path.join(DATASETS_DIR, 'gpqa/gpqa_diamond.json')
  example_start_index: int | None = None
  example_end_index: int | None = None

  def load(self):
    with epath.Path(self.path).open('r') as f:
      examples = json.load(f)
    new_examples = []
    for i, example in enumerate(examples):
      # using the same keys as DeepScaleR
      new_examples.append({
          'question': example['Question'],
          'correct_answer': example['Correct Answer'],
          'incorrect_answer_1': example['Incorrect Answer 1'],
          'incorrect_answer_2': example['Incorrect Answer 2'],
          'incorrect_answer_3': example['Incorrect Answer 3'],
          'example_id': example['Record ID'],
          'uid': f'gpqa_diamond-{i}',
          'id': i,
      })
    return new_examples[self.example_start_index:self.example_end_index]


class Dataloader:
  """Dataloader."""

  def __iter__(self):
    ...


class SimpleDataloader(Dataloader):
  """Simple dataloader."""

  def __init__(
      self, datasource: SimpleDataSource, batch_size: int, shuffle: bool = True,
      num_epochs: int | None = None, seed: int | None = None,
      num_past_examples: int = 0,
      drop_remainder: bool = True):
    assert batch_size > 0
    self.datasource = datasource
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.seed = seed
    self.num_past_examples = num_past_examples
    self.num_epochs = num_epochs
    self.drop_remainder = drop_remainder
    self._start_new_epoch()

  @property
  def _cursor(self):
    return self.num_past_examples % len(self.datasource)

  @property
  def current_epoch(self):
    return self.num_past_examples // len(self.datasource)

  def get_state(self) -> Mapping[str, Any]:
    return dict(num_past_examples=self.num_past_examples)

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.num_past_examples = state['num_past_examples']
    self._start_new_epoch()

  def _start_new_epoch(self):
    self.indices = list(range(len(self.datasource)))
    if self.shuffle:
      # Hacky way to create a deterministic shuffle for each epoch that is also
      # quick to recover from `num_past_examples`.
      rng = random.Random(self.seed + self.current_epoch)
      rng.shuffle(self.indices)

  def __iter__(self):
    while self.num_epochs is None or self.current_epoch < self.num_epochs:
      batch = []
      # Stop if the remaining examples in the current epoch are less than the
      # batch size when `drop_remainder` is True, otherwise the batch will be
      # filled with some examples coming from the next epoch.
      if (self.num_epochs and
          (self.num_past_examples + self.batch_size >
           len(self.datasource) * self.num_epochs) and
          self.drop_remainder):
        break
      for _ in range(self.batch_size):
        batch.append(self.datasource[self.indices[self._cursor]])
        self.num_past_examples += 1
        if self._cursor == 0: self._start_new_epoch()
      yield batch

  def repeat(self, num_repeat=1):
    if self.num_epochs is None:
      num_epochs = None
    else:
      num_epochs = self.num_epochs * num_repeat
    return SimpleDataloader(
        datasource=self.datasource, batch_size=self.batch_size,
        shuffle=self.shuffle, num_epochs=num_epochs, seed=self.seed,
        num_past_examples=self.num_past_examples,
        drop_remainder=self.drop_remainder)


def create_simple_dataset(
    name: str, batch_size: int, seed: int, shuffle: bool, num_epochs: int | None
):
  datasource = DataSourceRegistry.get_instance(name)
  data = datasource.load()
  dataset = SimpleDataloader(
      datasource=data,
      batch_size=batch_size,
      shuffle=shuffle,
      num_epochs=num_epochs,
      seed=seed,
      drop_remainder=False,
  )
  return dataset


class Dataset:
  """A wrapper of tf.data.Dataset to add processors with numpy and jax."""

  def __init__(
      self, tf_dataset: tf.data.Dataset,
      processors: Optional[list[Processor]] = None):
    self._tf_dataset = tf_dataset
    if processors is None:
      processors = []
    self._processors = processors

  def add_processor(self, processor: Processor):
    self._processors.append(processor)
    return self._processors

  def repeat(self, num_repeat):
    return self.copy(tf_dataset=self._tf_dataset.repeat(num_repeat))

  def copy(self, tf_dataset: Optional[tf.data.Dataset] = None,
           processors: Optional[list[Processor]] = None):
    if tf_dataset is None:
      tf_dataset = self._tf_dataset
    if processors is None:
      processors = self._processors
    return Dataset(tf_dataset, processors)

  def __iter__(self):
    def generator():
      for batch in self._tf_dataset.as_numpy_iterator():
        for processor in self._processors:
          batch = processor(batch)
        yield batch
    return generator()


def get_local_batch_size(batch_size: int):
  if batch_size % jax.device_count() == 0:
    local_batch_size = batch_size // jax.process_count()
  else:
    raise ValueError(f'Batch size {batch_size} must be divisible'
                     f' by total number of cores {jax.device_count()}.')
  return local_batch_size


def create_dataset_split(
    config, num_past_examples: int = 0, training: bool = True
):
  if config.feature_converter_name == 'LMFeatureConverter':
    task_feature_lengths = {'targets': config.seq_len}
    feature_converter_kwargs = {
        'pack': config.use_packing
    }
    if hasattr(config, 'bos_id'):
      feature_converter_kwargs['bos_id'] = config.bos_id
    feature_converter = seqio.LMFeatureConverter(**feature_converter_kwargs)
  elif config.feature_converter_name == 'PrefixLMFeatureConverter':
    task_feature_lengths = {
        'inputs': config.seq_len // 2, 'targets': config.seq_len // 2}
    feature_converter = seqio.PrefixLMFeatureConverter(
        pack=config.use_packing,
    )
  else:
    raise ValueError(
        f'Unsupported feature converter type: {config.feature_converter_name}'
    )
  dataset_name = config.dataset_name
  if not training and config.validation_dataset_name:
    dataset_name = config.validation_dataset_name
  if ':' in dataset_name:
    dataset_spec, dataset_name = dataset_name.split(':', maxsplit=1)
  else:
    dataset_spec = ''

  batch_size = config.batch_size
  if not training and config.validation_eval_batch_size > 0:
    batch_size = config.validation_eval_batch_size

  num_epochs = None
  if not training:
    num_epochs = config.validation_eval_epochs

  if dataset_spec == 'simply_json':
    return create_simple_dataset(
        name=f'{dataset_spec}:{dataset_name}',
        batch_size=batch_size,
        seed=config.dataset_seed,
        shuffle=training,
        num_epochs=num_epochs,
    )

  # TFDS datasets.
  dataset = seqio.get_dataset(
      f'{dataset_name}',
      task_feature_lengths=task_feature_lengths,
      dataset_split='train' if training else 'validation',
      shuffle=not training,
      num_epochs=num_epochs,
      use_cached=False,
      seed=config.dataset_seed,
      batch_size=batch_size,
      feature_converter=feature_converter,
  )
  return Dataset(dataset, [select_local_batch])


def create_dataset(config, num_past_examples: int = 0):
  train_set = create_dataset_split(config, num_past_examples, training=True)
  validation_set = None
  if config.use_validation_set:
    validation_set = create_dataset_split(
        config, num_past_examples, training=False
    )
  return train_set, validation_set


def select_local_batch(batch: Batch) -> Batch:
  """Selects the batch for the given process."""
  select_local_array_fn = functools.partial(
      select_local_array,
      process_index=jax.process_index(),
      num_processes=jax.process_count())
  new_batch = jax.tree_util.tree_map(select_local_array_fn, batch)
  return new_batch


def select_local_array(
    array: np.ndarray,
    process_index: int,
    num_processes: int) -> np.ndarray:
  """Selects the batch for the given process."""
  batch_size = array.shape[0]
  assert batch_size % num_processes == 0
  local_batch_size = batch_size // num_processes
  start_index = process_index * local_batch_size
  end_index = start_index + local_batch_size
  return array[start_index:end_index]


def create_chat_loss_mask(token_ids, mask_start_id, mask_end_id):
  def f(carry, a):
    new_carry = jnp.where(
        a == mask_end_id, -2, jnp.where(a == mask_start_id, -1, carry))
    return new_carry, carry

  token_ids = einops.rearrange(token_ids, 'b t -> t b')
  result = jax.lax.scan(f, jnp.full(token_ids.shape[1], -2), token_ids)[1] + 2
  return einops.rearrange(result, 't b -> b t')


def add_chat_loss_mask(batch, mask_start_id, mask_end_id):
  batch['decoder_loss_weights'] = create_chat_loss_mask(
      batch['decoder_target_tokens'], mask_start_id=mask_start_id,
      mask_end_id=mask_end_id) * batch['decoder_loss_weights']
  return batch
