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
	import type { Component } from 'svelte';
	import type { IconComponentProps } from 'phosphor-svelte';

	// Battery-style status chip used by the top-banner indicators (cpu, mem,
	// gcert, …). The pill has an internal fill bar whose width tracks `pct`,
	// with an optional icon + text overlaid. The tier governs color (border +
	// fill tint + text + icon — they all use currentColor).
	//
	// Semantics differ by signal but the visual convention is the same:
	//   more fill = more of the underlying quantity.
	// CPU/mem: fill = usage% (more fill = worse).
	// gcert:    fill = remaining/max (more fill = healthier).
	// Tier maps severity → color, so direction-of-badness is encoded in color
	// independently of the bar.
	//
	// pct=null renders an empty (unfilled) pill with dim text — used for the
	// "unknown" state before the first probe lands.

	export type Tier = 'ok' | 'low' | 'warning' | 'critical' | 'unknown' | 'hidden';
	export type IconType = Component<IconComponentProps>;

	let {
		tier,
		pct = null,
		text,
		tooltip = '',
		icon
	}: {
		tier: Tier;
		pct?: number | null; // 0-100, or null when unknown
		text: string;
		tooltip?: string;
		icon?: IconType; // optional phosphor-svelte icon; renders left of text
	} = $props();

	const clampedPct = $derived(pct === null ? 0 : Math.max(0, Math.min(100, pct)));
	const Icon = $derived(icon); // capitalize for the template's component renderer
</script>

{#if tier !== 'hidden'}
	<span class="chip tier-{tier}" title={tooltip}>
		<span class="fill" style="width: {clampedPct}%"></span>
		{#if Icon}<span class="icon"><Icon size={13} weight="bold" /></span>{/if}
		<span class="text">{text}</span>
	</span>
{/if}

<style>
	.chip {
		position: relative;
		display: inline-flex;
		align-items: center;
		/* No fixed width: with labels (CPU/RAM/GCERT) content has substance and
		   a fixed min would just bake in dead space on the right of shorter
		   pills. Bars are still proportional to each chip's own width, so visual
		   fullness compares cleanly chip-to-chip. */
		font-size: var(--fs-sm);
		padding: 0.18rem 0.55rem;
		border-radius: var(--radius-pill);
		border: 1px solid var(--border);
		color: var(--text-dim);
		font-family: var(--mono);
		font-variant-numeric: tabular-nums;
		overflow: hidden; /* clip the fill bar to the rounded shape */
		background: color-mix(in srgb, var(--bg-elev2) 60%, transparent);
	}
	.fill {
		position: absolute;
		top: 0;
		left: 0;
		bottom: 0;
		background: currentColor;
		opacity: 0.22; /* translucent so text overlay stays readable */
		/* Smooth bar movement on every push; brief enough to feel live. */
		transition:
			width 0.4s ease,
			background-color 0.2s ease,
			opacity 0.2s ease;
	}
	.icon {
		position: relative; /* sits above .fill */
		z-index: 1;
		display: inline-flex;
		align-items: center;
		margin-right: 0.35rem;
	}
	.text {
		position: relative; /* sits above .fill */
		z-index: 1;
		/* Explicit override (redundant with .chip's font-family but defensive):
		   guarantees the label + number always render in the mono stack
		   regardless of any future cascade rule. Mono matters here because the
		   numeric values churn (40% → 41% → 42% …) and a proportional font
		   would jitter the chip width on each update. */
		font-family: var(--mono);
		font-variant-numeric: tabular-nums;
	}
	/* Tier colors. .fill inherits currentColor; tier rules just set color
	   (text) and border, and let .fill follow. */
	.tier-ok {
		color: var(--gauge-ok);
		border-color: color-mix(in srgb, var(--gauge-ok) 35%, transparent);
	}
	.tier-unknown {
		color: var(--text-dim);
	}
	.tier-unknown .fill {
		opacity: 0;
	}
	.tier-low {
		color: var(--gauge-low);
		border-color: color-mix(in srgb, var(--gauge-low) 40%, transparent);
	}
	.tier-warning {
		color: var(--gauge-warning);
		border-color: color-mix(in srgb, var(--gauge-warning) 55%, transparent);
	}
	.tier-critical {
		color: var(--gauge-critical);
		border-color: color-mix(in srgb, var(--gauge-critical) 60%, transparent);
	}
	.tier-critical .fill {
		opacity: 0.32;
		animation: pulse 1.4s ease-in-out infinite;
	}
	@keyframes pulse {
		50% {
			opacity: 0.15;
		}
	}
</style>
