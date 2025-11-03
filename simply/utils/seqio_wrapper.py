"""PyGrain wrapper for SeqIO datasets."""

from collections.abc import Mapping
import dataclasses
import functools
from typing import Any, Self

import grain.python as grain
import seqio
import tensorflow.compat.v2 as tf


@dataclasses.dataclass(frozen=True)
class SeqIOConfig:
  """Configuration for SeqIO dataset."""

  dataset_name: str
  feature_converter_name: str
  batch_size: int
  seq_len: int
  split: str = 'train'
  use_packing: bool = True
  bos_id: int = 0

  use_cached: bool = True
  shuffle: bool = False
  num_epochs: int = 1
  seed: int | None = None

  def __post_init__(self):
    if self.use_cached:
      if self.shuffle:
        raise ValueError('shuffle=True is not supported with use_cached=True.')
      if self.num_epochs != 1:
        raise ValueError(
            'num_epochs != 1 is not supported with use_cached=True.'
        )
      if self.seed is not None:
        raise ValueError('seed is not supported with use_cached=True.')

  @functools.cached_property
  def task_feature_lengths(self) -> Mapping[str, int]:
    if self.feature_converter_name == 'LMFeatureConverter':
      return {'targets': self.seq_len}
    elif self.feature_converter_name == 'PrefixLMFeatureConverter':
      return {'inputs': self.seq_len // 2, 'targets': self.seq_len // 2}
    else:
      raise ValueError(
          f'Unsupported feature converter type: {self.feature_converter_name}'
      )

  @functools.cached_property
  def feature_converter(self) -> seqio.FeatureConverter:
    if self.feature_converter_name == 'LMFeatureConverter':
      return seqio.LMFeatureConverter(pack=self.use_packing, bos_id=self.bos_id)
    elif self.feature_converter_name == 'PrefixLMFeatureConverter':
      return seqio.PrefixLMFeatureConverter(pack=self.use_packing)
    else:
      raise ValueError(
          f'Unsupported feature converter type: {self.feature_converter_name}'
      )

  @classmethod
  def from_config(cls, config, **overrides) -> Self:
    kwargs = {
        field.name: getattr(config, field.name)
        for field in dataclasses.fields(cls)
        if hasattr(config, field.name)
    }
    kwargs.update(overrides)
    return cls(**kwargs)


def create_dataset(
    config: SeqIOConfig, worker_index: int, num_workers: int
) -> tf.data.Dataset:
  """Fork of simply/data_lib.py:create_dataset_split()."""
  dataset_name = config.dataset_name
  batch_size = config.batch_size
  shard_info = seqio.ShardInfo(index=worker_index, num_shards=num_workers)
  dataset = seqio.get_dataset(
      dataset_name,
      task_feature_lengths=config.task_feature_lengths,
      dataset_split=config.split,
      shuffle=config.shuffle,
      num_epochs=config.num_epochs,
      use_cached=config.use_cached,
      seed=config.seed,
      batch_size=batch_size,
      shard_info=shard_info,
      feature_converter=config.feature_converter,
  )
  # Here we disable autotune and multithreading in tf.data, because we will use
  # multiprocessing in PyGrain.
  options = tf.data.Options()
  options.autotune.enabled = False
  options.threading.max_intra_op_parallelism = 1
  options.threading.private_threadpool_size = 1
  dataset = dataset.with_options(options)
  return dataset


class _SeqIOIterator(grain.DatasetIterator[dict[str, Any]]):
  """Iterator that batches elements with a given batch function."""

  def __init__(self, config: SeqIOConfig, worker_index: int, num_workers: int):
    super().__init__()
    self._config = config
    self._worker_index = worker_index
    self._num_workers = num_workers
    self._example_counter = 0
    self._seqio_dataiter = None

  def __next__(self) -> dict[str, Any]:
    if self._seqio_dataiter is None:
      self._seqio_dataiter = create_dataset(
          self._config, self._worker_index, self._num_workers
      ).as_numpy_iterator()
      # Playback example_counter many examples to restore to proper state.
      for _ in range(self._example_counter):
        next(self._seqio_dataiter)

    self._example_counter += 1
    return next(self._seqio_dataiter)

  def get_state(self) -> Any:
    return {
        'config': dataclasses.asdict(self._config),
        'example_counter': self._example_counter,
    }

  def set_state(self, state) -> None:
    assert self._config == SeqIOConfig(**state['config'])
    self._example_counter = state['example_counter']
    self._seqio_dataiter = None

  def __str__(self) -> str:
    return (
        f'_SeqIOIterator(config={self._config},'
        f' example_counter={self._example_counter})'
    )


class SeqIODataset(grain.IterDataset[dict[str, Any]]):
  """Batch transformation for IterDatasets, using a batch function."""

  def __init__(self, config: SeqIOConfig):
    super().__init__()
    self._config = config
    self._num_workers = 1
    self._worker_index = 0

  def set_slice(self, sl: slice, sequential_slice: bool = False) -> None:
    del sequential_slice  # Unused.
    # This is the function internally called by mp_prefetch() to set the proper
    # data slice for each worker; it's not public PyGrain API, so no guarantee
    # going forward...
    assert sl.stop is None, f'{sl=}'
    self._num_workers = sl.step
    self._worker_index = sl.start

  def __iter__(self) -> _SeqIOIterator:
    return _SeqIOIterator(self._config, self._worker_index, self._num_workers)

  def __str__(self) -> str:
    return f'SeqIODataset(config={self._config})'


def make_train_data(
    config, num_workers: int = 32, worker_buffer_size: int = 2
) -> grain.IterDataset[dict[str, Any]]:
  """Returns a PyGrain dataset for training, from Simply experiment config."""
  config = SeqIOConfig.from_config(config)
  return SeqIODataset(config).mp_prefetch(
      grain.MultiprocessingOptions(
          num_workers=num_workers, per_worker_buffer_size=worker_buffer_size
      )
  )


def make_eval_data(
    config, num_workers: int = 32, worker_buffer_size: int = 2
) -> grain.IterDataset[dict[str, Any]]:
  """Returns a PyGrain dataset for evaluation, from Simply experiment config."""
  overrides = {'use_packing': False}
  if config.use_validation_set:
    overrides['split'] = 'validation'
    if config.validation_dataset_name is not None:
      overrides['dataset_name'] = config.validation_dataset_name
    if config.validation_eval_batch_size > 0:
      overrides['batch_size'] = config.validation_eval_batch_size

  config = SeqIOConfig.from_config(config, **overrides)
  return SeqIODataset(config).mp_prefetch(
      grain.MultiprocessingOptions(
          num_workers=num_workers, per_worker_buffer_size=worker_buffer_size
      )
  )
