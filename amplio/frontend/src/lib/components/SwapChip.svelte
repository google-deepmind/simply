<!--
 Copyright 2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

<script lang="ts">
	import { SwapIcon } from 'phosphor-svelte';
	import { sysstat } from '$lib/sysstat.svelte';
	import StatChip, { type Tier } from './StatChip.svelte';

	// Swap usage. Informational only — swap fullness is NOT the same as
	// swap pressure (the kernel parks cold pages here even when memory is
	// plentiful and may never page them back in). Real thrash = paging
	// activity (pswpin/pswpout rate), which we don't probe today. So the
	// tiering here is mild: yellow when notably full, never red — we don't
	// want to cry wolf on what's mostly a curiosity metric.
	//
	// Hidden when swap is disabled (probe returns null) or completely empty.

	const tier: Tier = $derived.by(() => {
		const s = sysstat.swapPct;
		if (s === null || s === 0) return 'hidden';
		if (s >= 80) return 'warning';
		if (s >= 50) return 'low';
		return 'ok';
	});

	const text = $derived(
		sysstat.swapPct === null ? 'SWAP —' : `SWAP ${Math.round(sysstat.swapPct)}%`
	);

	const tooltip =
		'Swap usage (currently stored). Not a thrash indicator — Linux parks ' +
		'cold pages in swap even when memory is plentiful.';
</script>

<StatChip {tier} pct={sysstat.swapPct} {text} {tooltip} icon={SwapIcon} />
