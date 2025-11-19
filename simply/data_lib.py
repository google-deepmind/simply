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
from typing import Callable, ClassVar, Mapping, MutableMapping, Optional, Protocol, Union

import einops
from etils import epath
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import seqio
from simply.utils import common
from simply.utils import registry
from simply.utils import seqio_wrapper
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
    path = os.path.join(DATASETS_DIR, f'redpajama_1t/tfrecord/{cat}.tfrecord*')
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


def create_simple_dataset(
    name: str, batch_size: int, seed: int, shuffle: bool, num_epochs: int | None
) -> grain.IterDataset[common.PyTree]:
  datasource = DataSourceRegistry.get_instance(name)
  data = datasource.load()
  dataset = grain.MapDataset.source(data)
  if shuffle:
    dataset = dataset.shuffle(seed=seed)
  return (
      dataset.repeat(num_epochs)
      .batch(batch_size, batch_fn=lambda x: x)
      .to_iter_dataset()
  )


# class TokenizeTransform(grain.MapTransform):
#   """Tokenizes text using a given tokenizer.

#   This is a custom transform that can use any tokenizer (e.g., SentencePiece).
#   """

#   def __init__(self, tokenizer, text_key='text', output_key='tokens',):
#     self.tokenizer = tokenizer
#     self.text_key = text_key
#     self.output_key = output_key

#   def map(self, features):
#     """Tokenize the text field."""
#     text = features[self.text_key]
#     if isinstance(text, bytes):
#       text = text.decode('utf-8')

#     # Tokenize using the provided tokenizer
#     # For demo purposes, we'll use a simple split - replace with actual tokenizer
#     tokens = self.tokenizer.encode(text)

#     # Update features with tokenized output
#     features = dict(features)  # Make a copy
#     del features[self.text_key]
#     features[self.output_key] = np.array(tokens, dtype=np.int32)
#     return features


# def create_grain_dataset(
#   data_source, tokenizer, batch_size=4, seed=0, seq_len=50,
#   # num_packing_bins=128,
#   # mode='concat_then_split',
#   add_eos=False, add_bos=False
#   ):
#   dataset = (
#       grain.MapDataset.source(data_source)
#       .shuffle(seed)
#       .repeat()
#       # Tokenize text
#       .map(TokenizeTransform(tokenizer, text_key='text', output_key='tokens'))
#   )
#   def add_bos_eos(x):
#     x = [tokenizer.bos_id] + x
#     x = x + [tokenizer.eos_id]
#     return x
#   if add_eos or add_bos:
#     dataset.map(lambda x: [tokenizer.bos_id] + x + [tokenizer.eos_id])
#   dataset = grain.experimental.ConcatThenSplitIterDataset(
#       parent=dataset,
#       length_struct={'tokens': seq_len},
#   )

#   # [bos, a, b, c, eos, bos, b, c, eos]
#   # [b, c, eos, bos, b,]
#   # [bos, b, c, eos, b, c, eos]
#   # # Apply FirstFit packing
#   # dataset = grain.experimental.FirstFitPackIterDataset(
#   #     parent=dataset,
#   #     length_struct={'tokens': seq_len},
#   #     num_packing_bins=num_packing_bins,
#   #     shuffle_bins=True,
#   # )

#   # Batch and prefetch
#   dataset = dataset.batch(batch_size, drop_remainder=True)
#   dataset = dataset.mp_prefetch(
#       grain.MultiprocessingOptions(num_workers=0, per_worker_buffer_size=10)
#   )
#   return dataset


def create_iter_dataset(
    config, training: bool = True
) -> grain.IterDataset[common.PyTree]:
  dataset_name = config.dataset_name
  batch_size = config.batch_size

  if training:
    split = 'train'
    shuffle = True
    num_epochs = None
  else:
    split = 'validation'
    if config.validation_dataset_name:
      dataset_name = config.validation_dataset_name
    if config.validation_eval_batch_size > 0:
      batch_size = config.validation_eval_batch_size
    shuffle = False
    num_epochs = config.validation_eval_epochs

  if dataset_name.startswith('simply_json:'):
    return create_simple_dataset(
        dataset_name, batch_size, config.dataset_seed, shuffle, num_epochs
    )

  if dataset_name.startswith('simply_det:'):
    seqio_config = seqio_wrapper.SeqIOConfig(
        dataset_name=dataset_name,
        feature_converter_name=config.feature_converter_name,
        batch_size=batch_size,
        seq_len=config.seq_len,
        split=split,
        use_packing=config.use_packing,
        bos_id=getattr(config, 'bos_id', 0),
        use_cached=True,
        shuffle=False,
        num_epochs=1,
        seed=None,
    )
  else:
    seqio_config = seqio_wrapper.SeqIOConfig(
        dataset_name=dataset_name,
        feature_converter_name=config.feature_converter_name,
        batch_size=batch_size,
        seq_len=config.seq_len,
        split=split,
        use_packing=False,
        bos_id=getattr(config, 'bos_id', 0),
        use_cached=False,
        shuffle=shuffle,
        num_epochs=num_epochs,
        seed=config.dataset_seed,
    )
  return seqio_wrapper.SeqIODataset(seqio_config).mp_prefetch(
      grain.MultiprocessingOptions(
          num_workers=config.prefetch_num_workers,
          per_worker_buffer_size=config.prefetch_per_worker_buffer_size,
      )
  )


def create_chat_loss_mask(token_ids, mask_start_id, mask_end_id):
  def f(carry, a):
    new_carry = jnp.where(
        a == mask_end_id, -2, jnp.where(a == mask_start_id, -1, carry)
    )
    return new_carry, carry

  token_ids = einops.rearrange(token_ids, 'b t -> t b')
  result = jax.lax.scan(f, jnp.full(token_ids.shape[1], -2), token_ids)[1] + 2
  return einops.rearrange(result, 't b -> b t')


def add_chat_loss_mask(batch, mask_start_id, mask_end_id):
  batch['decoder_loss_weights'] = create_chat_loss_mask(
      batch['decoder_target_tokens'], mask_start_id=mask_start_id,
      mask_end_id=mask_end_id) * batch['decoder_loss_weights']
  return batch
