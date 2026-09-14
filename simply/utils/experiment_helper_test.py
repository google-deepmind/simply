# Copyright 2024 The Simply Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for experiment_helper."""

from absl.testing import absltest
from absl.testing import parameterized
from simply.utils import experiment_helper


class MetricsAggregatorTest(parameterized.TestCase):

  def test_get_aggregated_metrics_default_mean(self):
    aggregator = experiment_helper.MetricsAggregator(average_last_n_steps=10)
    for val in [1.0, 2.0, 3.0, 4.0]:
      aggregator.add('loss', val)
    metrics = aggregator.get_aggregated_metrics()
    self.assertIn('loss', metrics)
    self.assertAlmostEqual(float(metrics['loss']), 2.5)

  def test_get_aggregated_metrics_median_odd(self):
    aggregator = experiment_helper.MetricsAggregator(average_last_n_steps=10)
    for val in [3.0, 1.0, 2.0]:
      aggregator.add('step_time', val)
    median_metrics = aggregator.get_aggregated_metrics(method='median')
    self.assertIn('step_time', median_metrics)
    self.assertAlmostEqual(float(median_metrics['step_time']), 2.0)

  def test_get_aggregated_metrics_median_even(self):
    aggregator = experiment_helper.MetricsAggregator(average_last_n_steps=10)
    for val in [1.0, 2.0, 3.0, 4.0]:
      aggregator.add('step_time', val)
    median_metrics = aggregator.get_aggregated_metrics(method='median')
    self.assertIn('step_time', median_metrics)
    self.assertAlmostEqual(float(median_metrics['step_time']), 2.5)

  def test_get_aggregated_metrics_invalid_method(self):
    aggregator = experiment_helper.MetricsAggregator(average_last_n_steps=10)
    aggregator.add('loss', 1.0)
    with self.assertRaises(ValueError):
      aggregator.get_aggregated_metrics(method='invalid')

  def test_median_robustness_against_outlier(self):
    aggregator = experiment_helper.MetricsAggregator(average_last_n_steps=10)
    for val in [0.1, 0.11, 0.09, 0.1, 1000.0]:
      aggregator.add('train_step_time', val)
    mean_metrics = aggregator.get_aggregated_metrics(method='mean')
    median_metrics = aggregator.get_aggregated_metrics(method='median')
    # Mean is corrupted by compile spike (~200.08)
    self.assertGreater(float(mean_metrics['train_step_time']), 100.0)
    # Median remains robust (~0.1)
    self.assertAlmostEqual(float(median_metrics['train_step_time']), 0.1)

  def test_experiment_helper_delegation(self):
    helper = experiment_helper.ExperimentHelper(
        experiment_dir='',
        metric_log_interval=10,
    )
    for val in [1.0, 3.0, 2.0]:
      helper.add_metric('step_time', val)
    mean_metrics = helper.get_aggregated_metrics(method='mean')
    median_metrics = helper.get_aggregated_metrics(method='median')
    self.assertAlmostEqual(float(mean_metrics['step_time']), 2.0)
    self.assertAlmostEqual(float(median_metrics['step_time']), 2.0)


if __name__ == '__main__':
  absltest.main()
