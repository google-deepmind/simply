"""File for pytest configuration."""

from absl import flags


def pytest_configure(config):
  del config
  flags.FLAGS.mark_as_parsed()
