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
import jax
import jax.numpy as jnp
from simply.utils import control_flow


ControlFlow = control_flow.ControlFlow
ControlStep = control_flow.ControlStep


class ControlFlowTest(absltest.TestCase):

  def test_assignment(self):
    f = ControlFlow(
        steps=[
            ControlStep(
                output_spec='y',
            ),
            ControlStep(
                fn='lambda x, y, f, g: (x, f, x + y + g)',
                overwrite_input_spec={'g': 'params/g/t[2]'},
                output_spec=('q', 'z/t[1]', 'z/t[2]'),
            ),
        ],
        output_spec={
            'a': 'q',
            'b': 'z'
        }
    )
    prng_key = jax.random.PRNGKey(0)
    f.init(prng_key)
    output = f.apply({'g': {'t': [1, 2, 3]}}, 1, f=10)
    self.assertEqual(output, {'a': 1, 'b': {'t': [None, 10, 5]}})

  def test_scan(self):
    f = control_flow.ScanModule(
        module=ControlFlow(
            steps=[
                ControlStep(
                    fn='lambda x, y, z: (x + y, x + y + z)',
                    output_spec=('x', 'u/v'),
                ),
            ],
            output_spec=('x', 'u'),
        ),
        length=2,
        per_step_args=['y'],
        overwrite_input_spec={'y': 'y', 'z': 'z'},
    )
    prng_key = jax.random.PRNGKey(0)
    params = f.init(prng_key)
    output, extra = f.apply(params, x=0.0, y=jnp.array([10, 20]), z=5)
    self.assertEqual(output, 30)
    self.assertEqual(
        jax.tree.map(lambda x: x.tolist(), extra), {'v': [15.0, 35.0]}
    )


if __name__ == '__main__':
  absltest.main()
