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

from unittest import mock

from absl.testing import absltest
from simply.utils import metric_writer
import tensorboardX


class MetricWriterTest(absltest.TestCase):

  def test_tensorboard_x_writer(self):
    with mock.patch.object(tensorboardX, 'SummaryWriter') as mock_writer_cls:
      mock_writer = mock_writer_cls.return_value
      writer = metric_writer.TensorboardXMetricWriter('/tmp/logdir')

      writer.write_scalars(1, {'loss': 0.1})
      mock_writer.add_scalar.assert_called_once_with('loss', 0.1, 1)

      writer.write_texts(1, {'config': 'text'})
      mock_writer.add_text.assert_called_once_with('config', 'text', 1)

      writer.flush()
      mock_writer.flush.assert_called_once()

      writer.close()
      mock_writer.close.assert_called_once()

  def test_create_metric_writer(self):
    # This test depends on environment, so we just check it returns an instance
    with mock.patch.object(tensorboardX, 'SummaryWriter'):
      # We mock SummaryWriter to avoid actual file creation if it picks TBX
      # Try mocking create_default_writer too if we pick CLU
      if metric_writer._HAS_CLU:
        with mock.patch.object(
            metric_writer.metric_writers, 'create_default_writer'
        ):
          writer = metric_writer.create_metric_writer('/tmp/logdir')
      else:
        writer = metric_writer.create_metric_writer('/tmp/logdir')

    self.assertIsInstance(writer, metric_writer.BaseMetricWriter)


if __name__ == '__main__':
  absltest.main()
