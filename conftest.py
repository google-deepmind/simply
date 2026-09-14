"""Pytest configuration for the Simply test suite.

Tests use `absltest.TestCase` helpers such as `create_tempdir()`, which read
absl flags. Under `absltest.main()` the flags are parsed for us; under pytest
nothing parses them, so a test module run on its own fails with
`UnparsedFlagAccessError`. Parse them once here, ignoring pytest's own argv.
"""

import sys

from absl import flags
import pytest


def pytest_configure(config):
  del config
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(sys.argv[:1], known_only=True)


@pytest.fixture(autouse=True, scope='module')
def restore_jax_mesh():
  """Undoes global `jax.set_mesh()` calls leaking from one test module to the next.

  Several test modules set a process-global mesh (`jax.set_mesh(mesh)` outside
  a `with`). Each test target runs in its own process in the internal build, so
  the leak is invisible there; a single pytest session shares one process.
  """
  import jax  # Local: keep pytest startup free of the JAX import.

  before = jax.sharding.get_mesh()
  yield
  if jax.sharding.get_mesh() != before:
    jax.sharding.set_mesh(before if before.axis_names else None)
