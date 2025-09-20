from absl.testing import absltest
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


if __name__ == "__main__":
  absltest.main()
