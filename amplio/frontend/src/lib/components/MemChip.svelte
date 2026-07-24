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
	import { MemoryIcon } from 'phosphor-svelte';
	import { sysstat } from '$lib/sysstat.svelte';
	import StatChip, { type Tier } from './StatChip.svelte';

	// Memory pressure. Tier is driven purely by mem%; swap shows in the
	// tooltip for context but does NOT escalate the tier — `swap_pct` is just
	// "what's currently stored in swap", which is mostly cold/idle pages on a
	// healthy system. The real thrash signal is paging activity (pswpin/
	// pswpout rate from /proc/vmstat), a separate signal we don't probe yet.

	const tier: Tier = $derived.by(() => {
		const m = sysstat.memPct;
		if (m === null) return 'unknown';
		if (m >= 95) return 'critical';
		if (m >= 85) return 'warning';
		if (m >= 70) return 'low';
		return 'ok';
	});

	const text = $derived(sysstat.memPct === null ? 'RAM —' : `RAM ${Math.round(sysstat.memPct)}%`);

	const tooltip = $derived(
		sysstat.memPct === null
			? 'memory usage unavailable'
			: `memory ${sysstat.memPct.toFixed(1)}%`
	);
</script>

<StatChip {tier} pct={sysstat.memPct} {text} {tooltip} icon={MemoryIcon} />
