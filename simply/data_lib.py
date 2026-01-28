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
"""Data pipeline for LLM training based on grain.

Usage:
  # In experiment config, set dataset_config to one of:

  # Option 1: String shorthand (looks up factory in DatasetRegistry)
  dataset_config = 'my_dataset_factory'

  # Option 2: TFDS dataset
  dataset_config = DatasetConfig(
    source=TFDSSourceConfig(name='c4:3.0.1', split='train[:90%]'),
  )

  # Option 3: Mixture with weights
  dataset_config = MixtureConfig(datasets=(
    (DatasetConfig(source='dataset_a'), 0.7),
    ('dataset_b', 0.3),  # String shorthand also works
  ))

  # Create iterator
  dataset = create_iter_dataset(experiment_config, training=True)
  for batch in dataset:
    # batch has: decoder_input_tokens, decoder_target_tokens,
    decoder_loss_weights
    ...
"""


from collections.abc import Mapping, Sequence
import dataclasses
import functools
import glob
import json
import os
from typing import Any, ClassVar

from absl import logging
from etils import epath
import grain.python as grain
import numpy as np
from simply.utils import common
from simply.utils import registry
from simply.utils import tokenization
import simply.utils.lm_format as lm_format_lib


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


tokenization.TokenizerRegistry.reset()


def register_vocabs():
  vocabs = (
      OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS +
      OPENMIX_V3_VOCABS + GEMMA2_VOCABS)
  for name, vocab_path in vocabs:
    # Use default argument to capture loop variables correctly.
    tokenization.TokenizerRegistry.register(
        lambda vocab_path=vocab_path: tokenization.SimplySentencePieceVocab(
            vocab_path), name=name)

register_vocabs()

tokenization.TokenizerRegistry.register(
        lambda: tokenization.SimplySentencePieceVocab(GEMMA3_VOCAB),
        name='vb262144_gemma3'
)


################################################################################
# Registries.
################################################################################


class DataSourceRegistry(registry.RootRegistry):
  """Registry for data sources (lazy-evaluated)."""

  namespace: ClassVar[str] = 'datasource_grain'


class DataConfigRegistry(registry.RootRegistry):
  """Registry for data configuration dataclasses."""

  namespace: ClassVar[str] = 'dataconfig_grain'


################################################################################
# Configuration dataclasses.
################################################################################


@DataConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class TFDSSourceConfig:
  """Configuration for a TensorFlow Datasets (TFDS) data source.

  Use this when you need to specify TFDS-specific options like split.

  Attributes:
    name: TFDS dataset name with optional version (e.g., 'lm1b:1.1.0', 'c4').
    split: Data split to use. Supports TFDS slicing syntax like 'train[:90%]',
      'train[10%:20%]', 'validation'.

  Example:
    tfds_config = TFDSSourceConfig(name='c4:3.0.1', split='train[:1000]')
    dataset_config = DatasetConfig(source=tfds_config)
  """

  name: str
  split: str = 'train'


@DataConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class HFDatasetSourceConfig:
  """Configuration for a HuggingFace Datasets data source.

  Use this for HuggingFace datasets from the Hub or local files.

  Attributes:
    name: Dataset name on HuggingFace Hub (e.g., 'imdb', 'wikitext').
    split: Data split to use (e.g., 'train', 'train[:1000]').
    subset: Optional subset/configuration name (e.g., 'wikitext-2-raw-v1').

  Example:
    hf_config = HFDatasetSourceConfig(name='imdb', split='train[:1000]')
    dataset_config = DatasetConfig(source=hf_config)

    # With subset/configuration
    hf_config = HFDatasetSourceConfig(
      name='wikitext',
      subset='wikitext-2-raw-v1',
      split='train',
    )
  """

  name: str
  split: str = 'train'
  subset: str | None = None


@DataConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class ArrayRecordSourceConfig:
  """Configuration for ArrayRecord data source.

  ArrayRecord is Google's efficient file format for random access to large
  datasets. See: https://github.com/google/array_record

  Attributes:
    paths: Sequence of file paths or glob patterns to ArrayRecord files.
      Examples: ('/data/train.arrayrecord',) or ('/data/train-*.arrayrecord',)

  Example:
    # Single file
    config = ArrayRecordSourceConfig(paths=('/data/train.arrayrecord',))

    # Multiple shards with glob
    config = ArrayRecordSourceConfig(
        paths=('/data/train-00000-of-00010.arrayrecord',
               '/data/train-00001-of-00010.arrayrecord',)
    )

    # In DatasetConfig
    dataset_config = DatasetConfig(
        source=ArrayRecordSourceConfig(paths=('/data/pile/*.arrayrecord',)),
        lm_format_name='Pretrain',
    )
  """

  paths: Sequence[str]


@DataConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class DatasetConfig:
  """Configuration for a single dataset.

  The data source can be specified in four ways:
  1. source_name (str): Looks up in DataSourceRegistry
  2. source (TFDSSourceConfig): Uses TFDS with specified name and split
  3. source (HFDatasetSourceConfig): Uses HuggingFace datasets
  4. source (ArrayRecordSourceConfig): Uses ArrayRecord files directly

  Attributes:
    source: Data source - either a string (registry name) or config object.
    lm_format_name: Controls tokenization/formatting:
      - None: Raw - skip tokenization, data passed through as-is (only
        batch/prefetch applied). Use for custom processing outside pipeline.
      - 'Pretrain': Pretraining - TokenizeTransform + NextTokenPredTransform.
      - 'SimplyV1Chat', etc.: Chat format - ChatFormatTransform with role
        markers + NextTokenPredTransform.
    packing: Packing method for creating fixed-length sequences:
      - 'concat_split': Concatenate and split (best throughput, for pretraining)
      - 'first_fit': Bin packing preserving boundaries (for chat/SFT)
      - 'pad_or_truncate': Simple pad/truncate (for validation)
      - 'none': No packing (for raw data)
    data_key: Key in the example dict containing data to process.
    tokenizer_name: Tokenizer to use from TokenizerRegistry. If None, uses the
      experiment's vocab_name.
    add_eos: Whether to append EOS token after tokenization. Only used for
      'Pretrain' format.
    add_bos: Whether to prepend BOS token after tokenization.
    trainable_roles: For chat formats, which roles should have loss computed.
      E.g., ('assistant',) means only assistant turns contribute to loss. If
      None, all roles contribute to loss.

  Example:
    # Pretraining with TFDS
    config = DatasetConfig(
      source=TFDSSourceConfig(name='c4:3.0.1', split='train[:90%]'),
      lm_format_name='Pretrain',
      packing='concat_split',
    )

    # Chat/SFT format with loss only on assistant turns
    config = DatasetConfig(
      source='chat_data',
      lm_format_name='SimplyV1Chat',
      packing='first_fit',
      data_key='conversation',
      trainable_roles=('assistant',),
    )

    # Raw format (custom processing outside pipeline)
    config = DatasetConfig(
      source='my_source',
      lm_format_name=None,
      packing='none',
    )
  """

  source: (
      str
      | TFDSSourceConfig
      | HFDatasetSourceConfig
      | ArrayRecordSourceConfig
  )
  lm_format_name: str | None = 'Pretrain'  # None = raw, 'Pretrain' = tokenize
  packing: str = 'concat_split'
  data_key: str = 'text'
  tokenizer_name: str | None = None  # None = use config.vocab_name
  add_eos: bool = True
  add_bos: bool = False
  trainable_roles: tuple[str, ...] | None = None  # None = all roles have loss.


@DataConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class MixtureConfig:
  """Configuration for a mixture of datasets.

  Supports two mixing strategies controlled by pack_before_mix:
  - pack_before_mix=False (default): Mix examples first, then pack into
    fixed-length sequences. Packed sequences may contain examples from
    different datasets.
  - pack_before_mix=True: Pack each dataset separately first, then mix
    the packed sequences. Each packed sequence contains examples from
    only one dataset.

  Attributes:
    datasets: Sequence of (DatasetConfig, weight) pairs. Weights are
      automatically normalized to sum to 1.0.
    pack_before_mix: If True, pack each dataset before mixing. If False
      (default), mix examples first then pack together.

  Example:
    # Mix then pack (default) - packed sequences may span datasets.
    mixture = MixtureConfig(
      datasets=(
        (DatasetConfig(source='dataset_a'), 0.7),
        (DatasetConfig(source='dataset_b'), 0.3),
      ),
    )

    # Pack then mix - each packed sequence from single dataset.
    mixture = MixtureConfig(
      datasets=(
        (DatasetConfig(source='dataset_a'), 0.7),
        (DatasetConfig(source='dataset_b'), 0.3),
      ),
      pack_before_mix=True,
    )
  """

  datasets: Sequence[tuple[DatasetConfig, float]]
  pack_before_mix: bool = False

  def __post_init__(self):
    if not self.datasets:
      raise ValueError('MixtureConfig requires at least one dataset')
    for _, weight in self.datasets:
      if weight <= 0:
        raise ValueError(f'Weight must be positive, got {weight}')


################################################################################
# JSON data sources (for evaluation/RL).
################################################################################


@functools.partial(DataSourceRegistry.register, name='simply_json:gsm8k_train')
@dataclasses.dataclass(frozen=True)
class GSM8KSource:
  """GSM8K dataset source with lazy loading."""

  path: str = os.path.join(DATASETS_DIR, 'gsm8k/gsm8k.json')
  split: str = 'train'
  start_index: int | None = None
  end_index: int | None = None

  @functools.cached_property
  def _examples(self) -> list[dict[str, Any]]:
    """Lazily load and cache examples."""
    with epath.Path(self.path).open('r') as f:
      data = json.load(f)
    examples = data[self.split]
    for i, example in enumerate(examples):
      example['uid'] = f'gsm8k_{self.split}-{i}'
      example['id'] = i
    return examples[self.start_index : self.end_index]

  def __len__(self) -> int:
    return len(self._examples)

  def __getitem__(self, index: int) -> dict[str, Any]:
    return self._examples[index]


@functools.partial(DataSourceRegistry.register, name='simply_json:gsm8k_test')
@dataclasses.dataclass(frozen=True)
class GSM8KTestSource(GSM8KSource):
  """GSM8K test split."""

  split: str = 'test'


@functools.partial(
    DataSourceRegistry.register, name='simply_json:simple_qa_test'
)
@dataclasses.dataclass(frozen=True)
class SimpleQASource:
  """Simple QA dataset source with lazy loading."""

  path: str = os.path.join(DATASETS_DIR, 'simple_qa/simple_qa_test_set.json')
  split: str = 'test'

  @functools.cached_property
  def _examples(self) -> list[dict[str, Any]]:
    with epath.Path(self.path).open('r') as f:
      data = json.load(f)
    examples = data[self.split]
    for i, example in enumerate(examples):
      example['uid'] = f'simple_qa_{self.split}-{i}'
      example['id'] = i
    return examples

  def __len__(self) -> int:
    return len(self._examples)

  def __getitem__(self, index: int) -> dict[str, Any]:
    return self._examples[index]


@functools.partial(DataSourceRegistry.register, name='simply_json:simple_qa_num')
@dataclasses.dataclass(frozen=True)
class SimpleQANumSource(SimpleQASource):
  """Simple QA dataset with only number-only answers."""

  path: str = os.path.join(
      DATASETS_DIR, 'simple_qa/simple_qa_test_set_number_only.json'
  )


@functools.partial(DataSourceRegistry.register, name='simply_json:mmlu_test')
@dataclasses.dataclass(frozen=True)
class MMLUSource:
  """MMLU dataset source with lazy loading."""

  path: str = os.path.join(DATASETS_DIR, 'mmlu/mmlu.json')
  split: str = 'test'
  start_index: int | None = None
  end_index: int | None = None

  @functools.cached_property
  def _examples(self) -> list[dict[str, Any]]:
    with epath.Path(self.path).open('r') as f:
      data = json.load(f)
    examples = data['data'][self.split]
    for i, example in enumerate(examples):
      example['uid'] = f'mmlu_{self.split}-{i}'
      example['id'] = i
    return examples[self.start_index : self.end_index]

  def __len__(self) -> int:
    return len(self._examples)

  def __getitem__(self, index: int) -> dict[str, Any]:
    return self._examples[index]


@functools.partial(DataSourceRegistry.register, name='simply_json:dsr40k_train')
@dataclasses.dataclass(frozen=True)
class DeepScaleRSource:
  """DeepScaleR dataset source with lazy loading."""

  path: str = os.path.join(DATASETS_DIR, 'deepscaler/deepscaler.json')
  start_index: int | None = None
  end_index: int | None = None

  @functools.cached_property
  def _examples(self) -> list[dict[str, Any]]:
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
    return new_examples[self.start_index : self.end_index]

  def __len__(self) -> int:
    return len(self._examples)

  def __getitem__(self, index: int) -> dict[str, Any]:
    return self._examples[index]


@functools.partial(DataSourceRegistry.register, name='simply_json:aime24')
@dataclasses.dataclass(frozen=True)
class AIME24Source:
  """AIME 2024 dataset source with lazy loading."""

  path: str = os.path.join(DATASETS_DIR, 'aime/aime_v2.json')
  year: int = 2024
  start_index: int | None = None
  end_index: int | None = None

  @functools.cached_property
  def _examples(self) -> list[dict[str, Any]]:
    with epath.Path(self.path).open('r') as f:
      examples = json.load(f)
    new_examples = []
    for i, example in enumerate(examples):
      if int(example['year']) == self.year:
        new_examples.append({
            'question': example['problem'],
            'short_answer': example['answer'],
            'answer': example['solution'],
            'uid': f'aime{self.year % 100}-{i}',
            'id': i,
        })
    return new_examples[self.start_index : self.end_index]

  def __len__(self) -> int:
    return len(self._examples)

  def __getitem__(self, index: int) -> dict[str, Any]:
    return self._examples[index]


@functools.partial(DataSourceRegistry.register, name='simply_json:aime25')
@dataclasses.dataclass(frozen=True)
class AIME25Source(AIME24Source):
  """AIME 2025 dataset source."""

  year: int = 2025


@functools.partial(DataSourceRegistry.register, name='simply_json:math500_test')
@dataclasses.dataclass(frozen=True)
class MATH500Source:
  """MATH500 test set source with lazy loading."""

  path: str = os.path.join(DATASETS_DIR, 'math500/test.json')
  start_index: int | None = None
  end_index: int | None = None

  @functools.cached_property
  def _examples(self) -> list[dict[str, Any]]:
    with epath.Path(self.path).open('r') as f:
      examples = json.load(f)
    new_examples = []
    for i, example in enumerate(examples):
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
    return new_examples[self.start_index : self.end_index]

  def __len__(self) -> int:
    return len(self._examples)

  def __getitem__(self, index: int) -> dict[str, Any]:
    return self._examples[index]


@functools.partial(DataSourceRegistry.register, name='simply_json:gpqa_diamond')
@dataclasses.dataclass(frozen=True)
class GPQADiamondSource:
  """GPQA-Diamond dataset source with lazy loading."""

  path: str = os.path.join(DATASETS_DIR, 'gpqa/gpqa_diamond.json')
  start_index: int | None = None
  end_index: int | None = None

  @functools.cached_property
  def _examples(self) -> list[dict[str, Any]]:
    with epath.Path(self.path).open('r') as f:
      examples = json.load(f)
    new_examples = []
    for i, example in enumerate(examples):
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
    return new_examples[self.start_index : self.end_index]

  def __len__(self) -> int:
    return len(self._examples)

  def __getitem__(self, index: int) -> dict[str, Any]:
    return self._examples[index]


def _register_gsm8k_variants():
  """Register GSM8K variants with limited examples."""
  for num_examples in [4, 32, 128]:
    DataSourceRegistry.register_value(
        GSM8KSource(start_index=0, end_index=num_examples),
        name=f'simply_json:gsm8k_train{num_examples}',
    )


_register_gsm8k_variants()


################################################################################
# Grain transforms.
################################################################################


@functools.cache
def _get_tokenizer(tokenizer_name: str):
  """Get tokenizer instance from TokenizerRegistry (cached)."""
  return tokenization.TokenizerRegistry.get_instance(tokenizer_name)


@functools.cache
def _get_lm_format(lm_format_name: str):
  """Get LMFormat instance from LMFormatRegistry (cached)."""
  return lm_format_lib.LMFormatRegistry.get_instance(lm_format_name)


@dataclasses.dataclass(frozen=True)
class TFExampleDeserializeTransform(grain.MapTransform):
  """Conditionally deserializes TFExample proto bytes to a dict.

  ArrayRecord files often contain serialized TFExample protos. This transform
  checks the input format at runtime:
  - If input is bytes: parses as TFExample proto and extracts features
  - If input is already a dict/Mapping: passes through unchanged

  This allows the transform to be applied unconditionally in the pipeline,
  handling both serialized and non-serialized sources gracefully.

  Supports bytes_list, int64_list, and float_list TFExample features.

  Example:
    # Safe to apply even if source may return dicts
    ds = ds.map(TFExampleDeserializeTransform())
    ds = ds.map(TokenizeTransform(tokenizer_name='my_tokenizer'))
  """

  def map(self, features: bytes | Mapping[str, Any]) -> dict[str, Any]:
    # Pass through if already a dict (e.g., HuggingFace, TFDS sources).
    if isinstance(features, Mapping):
      return dict(features)

    # Parse TFExample proto using tensorflow_datasets proto definitions.
    # This avoids requiring full tensorflow installation.
    # pylint: disable=g-import-not-at-top
    from tensorflow_datasets.proto import tf_example_pb2

    example = tf_example_pb2.Example()
    example.ParseFromString(features)

    result = {}
    for key, feature in example.features.feature.items():
      if feature.HasField('bytes_list'):
        values = list(feature.bytes_list.value)
        result[key] = values[0] if len(values) == 1 else values
      elif feature.HasField('int64_list'):
        values = list(feature.int64_list.value)
        result[key] = values[0] if len(values) == 1 else values
      elif feature.HasField('float_list'):
        values = list(feature.float_list.value)
        result[key] = values[0] if len(values) == 1 else values

    return result


@dataclasses.dataclass(frozen=True)
class TokenizeTransform(grain.MapTransform):
  """Tokenizes text using a tokenizer from TokenizerRegistry.

  This transform reads text from a specified key, tokenizes it, and outputs
  a 'tokens' array. Optionally adds BOS/EOS tokens.
  """

  tokenizer_name: str
  data_key: str = 'text'
  add_eos: bool = True
  add_bos: bool = False

  def map(self, features: Mapping[str, Any]) -> dict[str, Any]:
    text = features[self.data_key]
    if isinstance(text, bytes):
      text = text.decode('utf-8')

    tokenizer = _get_tokenizer(self.tokenizer_name)
    tokens = list(tokenizer.encode(text))

    if self.add_bos and tokenizer.bos_id is not None:
      tokens = [tokenizer.bos_id] + tokens
    if self.add_eos and tokenizer.eos_id is not None:
      tokens = tokens + [tokenizer.eos_id]

    return {'tokens': np.array(tokens, dtype=np.int32)}


@dataclasses.dataclass(frozen=True)
class NextTokenPredTransform(grain.MapTransform):
  """Converts tokens to next-token prediction format for decoder-only LMs.

  Input: {'tokens': [t1, t2, ..., tn], 'token_loss_mask': [...] (optional)}
  Output: {
    'decoder_input_tokens': [t1, t2, ..., tn-1],
    'decoder_target_tokens': [t2, t3, ..., tn],
    'decoder_loss_weights': [w2, w3, ..., wn],
  }

  If 'token_loss_mask' is present in features (from ChatFormatTransform),
  it will be used as the loss weights (shifted to align with targets).
  Otherwise, all weights are 1.0.
  """

  def map(self, features: Mapping[str, Any]) -> dict[str, Any]:
    tokens = features['tokens']

    # Input: [t1, t2, ..., tn-1] (drop last token)
    decoder_input = tokens[:-1]
    # Target: [t2, t3, ..., tn] (drop first token)
    decoder_target = tokens[1:]
    seq_len = len(decoder_target)

    # Loss weights: use token_loss_mask if available, otherwise all 1s.
    if 'token_loss_mask' in features:
      # Shift mask to align with targets (drop first element).
      loss_weights = features['token_loss_mask'][1:]
    else:
      loss_weights = np.ones(seq_len, dtype=np.float32)

    return {
        'decoder_input_tokens': decoder_input.astype(np.int32),
        'decoder_target_tokens': decoder_target.astype(np.int32),
        'decoder_loss_weights': loss_weights.astype(np.float32),
    }


@dataclasses.dataclass(frozen=True)
class ChatFormatTransform(grain.MapTransform):
  """Formats chat conversations with role tokens and tokenizes.

  Uses LMFormat from LMFormatRegistry for chat formatting tokens. This ensures
  consistency with inference/sampling code.

  Expects input with a conversation field containing JSON-serialized
  list of messages with 'role' and 'content' keys.

  Outputs:
    - tokens: Tokenized conversation
    - token_loss_mask: Per-token loss mask (1.0 for trainable roles, 0.0
    otherwise)

  The token_loss_mask is computed by tracking which tokens belong to which
  role's content. Role markers are not trainable. End-of-message markers are
  trainable when the role is trainable (so the model learns when to stop).

  Note: Unlike TokenizeTransform, this does not have add_eos since LMFormat
  already provides end_of_message_marker after each turn.

  Example:
    # Only train on assistant responses using SimplyV1Chat format
    transform = ChatFormatTransform(
      tokenizer_name='my_tokenizer',
      lm_format_name='SimplyV1Chat',
      trainable_roles=('assistant',),
    )
  """

  tokenizer_name: str
  lm_format_name: str
  data_key: str = 'conversation'
  add_bos: bool = False
  trainable_roles: tuple[str, ...] | None = None  # None = all roles trainable.

  def map(self, features: Mapping[str, Any]) -> dict[str, Any]:
    conversation = features.get(self.data_key)
    if isinstance(conversation, bytes):
      conversation = conversation.decode('utf-8')

    messages = json.loads(conversation)

    # Build tokens and loss mask together.
    lm_fmt = _get_lm_format(self.lm_format_name)
    tokenizer = _get_tokenizer(self.tokenizer_name)
    tokens, loss_mask = lm_fmt.format_tokens(
        messages, tokenizer, self.trainable_roles
    )

    if self.add_bos and tokenizer.bos_id is not None:
      tokens = [tokenizer.bos_id] + tokens
      loss_mask = [0.0] + loss_mask  # BOS token not in loss.

    return {
        'tokens': np.array(tokens, dtype=np.int32),
        'token_loss_mask': np.array(loss_mask, dtype=np.float32),
    }


@dataclasses.dataclass(frozen=True)
class TruncateTransform(grain.MapTransform):
  """Truncates sequences to a maximum length from the left.

  Truncates all 1D numpy arrays in features to seq_len, keeping the end
  (most recent tokens). Shorter sequences are left unchanged.

  Left truncation is used because for validation/evaluation, we want to
  keep the most recent context and the response we're evaluating, rather
  than losing them to truncation.

  Can be composed with PadTransform for fixed-length output:
    ds.map(TruncateTransform(seq_len)).map(PadTransform(seq_len, pad_id))
  """

  seq_len: int

  def map(self, features: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in features.items():
      if isinstance(value, np.ndarray) and value.ndim == 1:
        # Left truncation: keep the end (most recent tokens).
        result[key] = value[-self.seq_len:]
      else:
        result[key] = value
    return result


@dataclasses.dataclass(frozen=True)
class PadTransform(grain.MapTransform):
  """Pads sequences to a fixed length.

  Pads token arrays (keys ending in '_tokens') with pad_id, and weight arrays
  (keys ending in '_weights') with 0.0. Sequences longer than seq_len are
  left unchanged (use TruncateTransform first if truncation is needed).

  Can be composed with TruncateTransform for fixed-length output:
    ds.map(TruncateTransform(seq_len)).map(PadTransform(seq_len, pad_id))
  """

  seq_len: int
  pad_id: int = 0

  def map(self, features: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in features.items():
      if key.endswith('_tokens'):
        result[key] = common.pad_to_len(
            value, self.seq_len, self.pad_id, np.int32)
      elif key.endswith('_weights'):
        result[key] = common.pad_to_len(value, self.seq_len, 0.0, np.float32)
      else:
        result[key] = value
    return result


################################################################################
# Packing methods.
################################################################################

# Packing method constants.
# Currently a string, but can be extended to a PackingConfig dataclass later
# for advanced options (e.g., num_packing_bins, shuffle_bins).
PACKING_CONCAT_SPLIT = 'concat_split'
PACKING_FIRST_FIT = 'first_fit'
PACKING_PAD_OR_TRUNCATE = 'pad_or_truncate'
PACKING_NONE = 'none'


def _to_fixed_length(
    dataset: grain.IterDataset,
    seq_len: int,
    packing_method: str,
    pad_id: int = 0,
    seed: int = 0,
    num_packing_bins: int = 64,
    shuffle_bins: bool = True,
) -> grain.IterDataset:
  """Applies packing to create fixed-length sequences from an IterDataset.

  Args:
    dataset: Input IterDataset with variable-length sequences.
    seq_len: Target sequence length.
    packing_method: Method for creating fixed-length sequences:
      - 'concat_split': Concatenate examples then split at seq_len boundaries.
        Most efficient for throughput, packed sequences may span examples.
      - 'first_fit': First-fit bin packing. Each packed sequence contains
        complete examples (up to seq_len). Better for evaluation metrics.
      - 'pad_or_truncate': Truncate long sequences, pad short ones.
        One example per sequence with fixed length.
      - 'none': No packing operations. Pass through as-is (for raw data
        that's already formatted).
    pad_id: Padding token ID (used for 'pad_or_truncate' method).
    seed: Random seed for deterministic packing (used for first_fit).
    num_packing_bins: Number of bins for first_fit packing.
    shuffle_bins: Whether to shuffle bins in first_fit packing.

  Returns:
    A grain.IterDataset with fixed-length sequences.

  Raises:
    ValueError: If packing_method is unknown.
  """
  length_struct = {
      'decoder_input_tokens': seq_len,
      'decoder_target_tokens': seq_len,
      'decoder_loss_weights': seq_len,
  }

  if packing_method == PACKING_CONCAT_SPLIT:
    return grain.experimental.ConcatThenSplitIterDataset(
        parent=dataset,
        length_struct=length_struct,
    )
  elif packing_method == PACKING_FIRST_FIT:
    # Truncate first to ensure examples fit in bins.
    truncated = dataset.map(TruncateTransform(seq_len))
    return grain.experimental.FirstFitPackIterDataset(
        parent=truncated,
        length_struct=length_struct,
        num_packing_bins=num_packing_bins,
        seed=seed,
        shuffle_bins=shuffle_bins,
    )
  elif packing_method == PACKING_PAD_OR_TRUNCATE:
    return (
        dataset.map(TruncateTransform(seq_len))
        .map(PadTransform(seq_len, pad_id=pad_id))
    )
  elif packing_method == PACKING_NONE:
    return dataset
  else:
    raise ValueError(
        f'Unknown packing_method: {packing_method}. Expected one of:'
        f' {PACKING_CONCAT_SPLIT}, {PACKING_FIRST_FIT},'
        f' {PACKING_PAD_OR_TRUNCATE}, {PACKING_NONE}'
    )


################################################################################
# Data source creation.
################################################################################


def _load_hf_dataset(name: str, subset: str | None, split: str):
  """Load a HuggingFace dataset. HF datasets support random access natively."""
  import datasets  # pylint: disable=g-import-not-at-top

  return datasets.load_dataset(name, name=subset, split=split)


def get_data_source(config: DatasetConfig) -> grain.MapDataset:
  """Creates a Grain MapDataset from config.

  The data source is determined by config.source:
  - If TFDSSourceConfig: Uses tensorflow_datasets
  - If HFDatasetSourceConfig: Uses HuggingFace datasets library
  - If str: Looks up in DataSourceRegistry

  Note: This function returns raw (un-tokenized) data. For factory functions
  in DatasetRegistry, the pipeline calls them directly to get tokenized data.

  Args:
    config: DatasetConfig specifying the data source.

  Returns:
    A grain.MapDataset that lazily loads data (un-tokenized).

  Raises:
    ValueError: If the data source cannot be found.
  """
  source = config.source

  # Handle TFDSSourceConfig.
  if isinstance(source, TFDSSourceConfig):
    import tensorflow_datasets as tfds  # pylint: disable=g-import-not-at-top
    # Try array_record format first (preferred for grain).
    tfds_source = tfds.data_source(source.name, split=source.split)
    return grain.MapDataset.source(tfds_source)

  # Handle HFDatasetSourceConfig.
  # HF datasets support random access natively, so pass directly to grain.
  if isinstance(source, HFDatasetSourceConfig):
    hf_dataset = _load_hf_dataset(source.name, source.subset, source.split)
    return grain.MapDataset.source(hf_dataset)

  # Handle ArrayRecordSourceConfig.
  if isinstance(source, ArrayRecordSourceConfig):
    # Expand glob patterns in paths.
    expanded_paths = []
    for pattern in source.paths:

      # matches = glob.glob(pattern)
      if matches:
        expanded_paths.extend(sorted(matches))
      else:
        # No glob match - use as literal path (will error if doesn't exist).
        expanded_paths.append(pattern)
    ar_source = grain.ArrayRecordDataSource(expanded_paths)
    return grain.MapDataset.source(ar_source)

  # Handle string: look up in DataSourceRegistry.
  if isinstance(source, str):
    if source in DataSourceRegistry.keys():
      datasource = DataSourceRegistry.get_instance(source)
      return grain.MapDataset.source(datasource)

    raise ValueError(
        f'Data source not found: {source}. Register it in DataSourceRegistry or'
        ' use TFDSSourceConfig/HFDatasetSourceConfig for external datasets.'
    )

  raise ValueError(
      f'Invalid source type: {type(source)}. Expected str, TFDSSourceConfig, '
      'or HFDatasetSourceConfig.'
  )


################################################################################
# Internal helpers.
################################################################################


def _create_map_dataset(
    ds_config: DatasetConfig,
    tokenizer_name: str,
    seed: int,
    shuffle: bool,
    num_epochs: int | None,
    seed_offset: int = 0,
) -> grain.MapDataset:
  """Process a single DatasetConfig to MapDataset (before packing).

  Handles the full processing pipeline:
    source -> tokenize -> format -> shuffle -> repeat.

  Tokenization is controlled by ds_config.lm_format_name:
    - None: Raw - skip tokenization
    - 'Pretrain': TokenizeTransform + NextTokenPredTransform
    - Other: ChatFormatTransform + NextTokenPredTransform

  Args:
    ds_config: DatasetConfig specifying the dataset.
    tokenizer_name: Default tokenizer name if not in ds_config.
    seed: Random seed for shuffling.
    shuffle: Whether to shuffle the dataset.
    num_epochs: Number of epochs to repeat (None = infinite).
    seed_offset: Offset to add to seed (for mixture sub-datasets).

  Returns:
    A grain.MapDataset ready for packing via _to_fixed_length().
  """
  tk_name = ds_config.tokenizer_name or tokenizer_name
  fmt_name = ds_config.lm_format_name

  # Step 1a: Get raw data source.
  ds = get_data_source(ds_config)

  # Step 1b: Deserialize TFExample protos if needed.
  # TFExampleDeserializeTransform handles both bytes (deserializes) and dicts
  # (passes through), so we always apply it for sources that may return bytes.
  # This is a no-op for sources already returning dicts.
  ds = ds.map(TFExampleDeserializeTransform())

  # Step 2: Apply tokenization based on lm_format_name.
  if fmt_name is None:
    # Raw: skip tokenization, data passed through as-is.
    pass
  elif fmt_name == 'Pretrain':
    # Pretraining: tokenize text directly.
    ds = ds.map(
        TokenizeTransform(
            tokenizer_name=tk_name,
            data_key=ds_config.data_key,
            add_eos=ds_config.add_eos,
            add_bos=ds_config.add_bos,
        )
    )
    ds = ds.map(NextTokenPredTransform())
  else:
    # Chat format: apply ChatFormatTransform.
    ds = ds.map(
        ChatFormatTransform(
            tokenizer_name=tk_name,
            lm_format_name=fmt_name,
            data_key=ds_config.data_key,
            add_bos=ds_config.add_bos,
            trainable_roles=ds_config.trainable_roles,
        )
    )
    ds = ds.map(NextTokenPredTransform())

  # Step 3: Shuffle and repeat.
  if shuffle:
    ds = ds.shuffle(seed=seed + seed_offset)
  ds = ds.repeat(num_epochs)

  return ds


################################################################################
# Main entry point.
################################################################################


def create_iter_dataset(
    config,
    training: bool = True,
) -> grain.IterDataset:
  """Main entry point for creating datasets.

  Args:
    config: ExperimentConfig with dataset configuration. Must have
      dataset_config and optionally validation_dataset_config.
    training: If True, creates training dataset; else validation dataset.

  Returns:
    A grain.IterDataset ready for iteration.
  """
  # Determine dataset config and parameters based on training mode.
  if training:
    ds_config = config.dataset_config
    batch_size = config.batch_size
    shuffle = True
    num_epochs = None
  else:
    ds_config = (
        getattr(config, 'validation_dataset_config', None)
        or config.dataset_config
    )
    batch_size = (
        config.validation_eval_batch_size
        if config.validation_eval_batch_size > 0
        else config.batch_size
    )
    shuffle = False
    num_epochs = config.validation_eval_epochs

  # Get config values.
  tokenizer_name = config.vocab_name
  seq_len = config.seq_len
  seed = config.dataset_seed
  prefetch_num_workers = config.prefetch_num_workers
  prefetch_per_worker_buffer_size = config.prefetch_per_worker_buffer_size

  # Get pad_id from tokenizer (used for pad_or_truncate packing).
  tokenizer = _get_tokenizer(tokenizer_name)
  pad_id = tokenizer.pad_id or 0

  # Helper to convert MapDataset to IterDataset via packing.
  def _pack(map_ds: grain.MapDataset, packing: str) -> grain.IterDataset:
    # For validation, override to pad_or_truncate.
    if not training and packing != PACKING_PAD_OR_TRUNCATE:
      logging.warning(
          'Validation mode: overriding packing=%r to %r.',
          packing,
          PACKING_PAD_OR_TRUNCATE,
      )
    packing_method = packing if training else PACKING_PAD_OR_TRUNCATE
    return _to_fixed_length(
        map_ds.to_iter_dataset(), seq_len, packing_method, pad_id, seed)

  # Helper to apply batching and prefetching.
  def _finalize(iter_ds: grain.IterDataset) -> grain.IterDataset:
    iter_ds = iter_ds.batch(batch_size, drop_remainder=True)
    return iter_ds.mp_prefetch(
        grain.MultiprocessingOptions(
            num_workers=prefetch_num_workers,
            per_worker_buffer_size=prefetch_per_worker_buffer_size,
        )
    )

  # MixtureConfig: create sub-datasets and mix.
  if isinstance(ds_config, MixtureConfig):
    datasets = []
    weights = []
    packings = []
    for i, (entry, weight) in enumerate(ds_config.datasets):
      ds = _create_map_dataset(
          entry,
          tokenizer_name,
          seed,
          shuffle,
          num_epochs,
          seed_offset=i,
      )
      if ds_config.pack_before_mix:
        packing_method = entry.packing if training else PACKING_PAD_OR_TRUNCATE
        ds = _to_fixed_length(
            ds.to_iter_dataset(), seq_len, packing_method, pad_id, seed)
      datasets.append(ds)
      weights.append(weight)
      packings.append(entry.packing)

    if ds_config.pack_before_mix:
      iter_ds = grain.IterDataset.mix(datasets, weights=weights)
    else:
      # For mix-then-pack, use the most conservative packing method.
      # If any dataset uses first_fit, use that to preserve example boundaries.
      mixed_packing = (
          PACKING_FIRST_FIT if PACKING_FIRST_FIT in packings
          else PACKING_CONCAT_SPLIT
      )
      iter_ds = _pack(
          grain.MapDataset.mix(datasets, weights=weights), mixed_packing)
    return _finalize(iter_ds)

  # Single DatasetConfig.
  map_ds = _create_map_dataset(
      ds_config,
      tokenizer_name,
      seed,
      shuffle,
      num_epochs,
  )
  return _finalize(_pack(map_ds, ds_config.packing))
