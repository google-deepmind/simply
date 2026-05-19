# Copyright 2026 The Simply Authors
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
"""Default XLA flags for Simply."""

from typing import Mapping


DEFAULT_DECODE_FLAGS: Mapping[str, str | int | float | bool | None] = {
    'xla_tpu_enable_async_all_to_all': 'true',
    'xla_enable_async_collective_permute': 'true',
    'xla_enable_async_all_gather': 'false',
    'xla_tpu_enable_all_experimental_scheduler_features': 'false',
    'xla_tpu_enable_net_router_in_all_gather': 'false',
    'xla_tpu_nd_short_transfer_max_chunks': 1,
    'xla_jf_spmd_threshold_for_windowed_einsum_mib': 100000,
}
