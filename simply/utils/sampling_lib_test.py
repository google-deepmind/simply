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

from absl.testing import absltest
import numpy as np
from simply.utils import pytree
from simply.utils import sampling_lib


class SamplingLibTest(absltest.TestCase):

  def test_decoding_schedule(self):
    schedule = sampling_lib.DecodingSchedule(
        prefill_size=105,
        begin_position=80,
        end_position=2000,
        chunk_size=100,
    )
    self.assertEqual(80, schedule.get_next_length(0))
    self.assertEqual(105, schedule.get_next_length(80))
    self.assertEqual(205, schedule.get_next_length(105))
    self.assertEqual(2000, schedule.get_next_length(1905))

  def test_sampling_params(self):
    sampling_params = sampling_lib.SamplingParams(
        intermediate_decode_steps=128,
        max_decode_steps=1000,
    )
    schedule = sampling_params.get_decoding_schedule(
        min_input_length=200, max_input_length=400
    )

    self.assertEqual(256, schedule.prefill_size)
    self.assertEqual(199, schedule.begin_position)
    self.assertEqual(1399, schedule.end_position)
    self.assertEqual(128, schedule.chunk_size)

  def test_processed_input_batching(self):
    input1 = sampling_lib.ProcessedInput(
        tokens=[1, 2],
        extra_inputs={"extra_field": np.ones((1, 3))},
    )
    input2 = sampling_lib.ProcessedInput(
        tokens=[1, 2, 3, 4],
        extra_inputs={"extra_field": np.ones((2, 2))},
    )
    input3 = sampling_lib.ProcessedInput(
        tokens=[1, 2, 3],
        extra_inputs={"extra_field": np.ones((3, 1))},
    )

    batch = sampling_lib.ProcessedInputBatch.from_unpadded_inputs(
        [input1, input2, input3]
    )

    expected_extra_field = np.array([
        [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
        [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
        [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
    ])

    np.testing.assert_equal(
        batch.tokens,
        np.array([
            [1, 2, 0, 0],
            [1, 2, 3, 4],
            [1, 2, 3, 0],
        ]),
    )
    np.testing.assert_equal(batch.lengths, np.array([2, 4, 3]))
    np.testing.assert_equal(
        batch.extra_inputs["extra_field"], expected_extra_field
    )

  def test_chunk_dump_and_load(self):
    chunk = sampling_lib.Chunk(
        type=sampling_lib.Chunk.Type.TEXT,
        content="chunk",
    )
    tmp_path = self.create_tempfile().full_path
    pytree.save_pytree_to(chunk, tmp_path)
    loaded = pytree.load_pytree_from(tmp_path)
    self.assertEqual(chunk, loaded)


if __name__ == "__main__":
  absltest.main()
