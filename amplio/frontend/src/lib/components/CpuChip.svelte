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
	import { CpuIcon } from 'phosphor-svelte';
	import { sysstat } from '$lib/sysstat.svelte';
	import StatChip, { type Tier } from './StatChip.svelte';

	// CPU busy %. Tooltip surfaces the 1-min load average too (different
	// signal: load includes IO-wait + runnable count).

	const tier: Tier = $derived.by(() => {
		const p = sysstat.cpuPct;
		if (p === null) return 'unknown';
		if (p < 50) return 'ok';
		if (p < 80) return 'low';
		if (p < 95) return 'warning';
		return 'critical';
	});

	const text = $derived(sysstat.cpuPct === null ? 'CPU —' : `CPU ${Math.round(sysstat.cpuPct)}%`);

	const tooltip = $derived.by(() => {
		const parts: string[] = [];
		if (sysstat.cpuPct !== null) parts.push(`CPU ${sysstat.cpuPct.toFixed(1)}%`);
		if (sysstat.loadAvg1m !== null) parts.push(`load (1m) ${sysstat.loadAvg1m.toFixed(2)}`);
		return parts.join(' · ') || 'CPU usage unavailable';
	});
</script>

<StatChip {tier} pct={sysstat.cpuPct} {text} {tooltip} icon={CpuIcon} />
