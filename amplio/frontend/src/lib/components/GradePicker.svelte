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

<script lang="ts" module>
	import {
		SmileyXEyesIcon,
		SmileyAngryIcon,
		SmileyMehIcon,
		SmileyIcon,
		SmileyWinkIcon,
		CircleDashedIcon
	} from 'phosphor-svelte';
	import type { Component } from 'svelte';

	// One grade rank's display metadata. `value` is the boundary string sent to
	// the API; `icon` is the phosphor smiley; `color` is the rank's accent (a
	// green→red ramp), used subtly for the icon tint.
	type GradeSpec = {
		value: string;
		label: string;
		icon: Component;
		color: string;
	};

	// Canonical grade table, best→worst (top reads as the aspirational top of the
	// scale in the dropdown). The integer rank is 5..1; index here is purely
	// presentational. Icons were verified to exist in phosphor-svelte:
	//   excellent → SmileyWink, good → Smiley, meh → SmileyMeh,
	//   bad → SmileyAngry, garbage → SmileyXEyes.
	// Colors lean on the theme's --ok (green) and --err (red), with neutral
	// amber/text-dim in the middle. color-mix keeps them muted so a list of
	// graded cards stays calm rather than a traffic-light wall.
	export const GRADE_SPECS: GradeSpec[] = [
		{ value: 'excellent', label: 'Excellent', icon: SmileyWinkIcon, color: 'var(--ok)' },
		{
			value: 'good',
			label: 'Good',
			icon: SmileyIcon,
			color: 'color-mix(in srgb, var(--ok) 70%, var(--warn))'
		},
		{ value: 'meh', label: 'Meh', icon: SmileyMehIcon, color: 'var(--warn)' },
		{
			value: 'bad',
			label: 'Bad',
			icon: SmileyAngryIcon,
			color: 'color-mix(in srgb, var(--err) 70%, var(--warn))'
		},
		{ value: 'garbage', label: 'Garbage', icon: SmileyXEyesIcon, color: 'var(--err)' }
	];

	// Lookup a spec by its boundary string. Returns undefined for null/unknown
	// (ungraded), which the UI renders as a dashed circle.
	export function gradeSpec(value: string | null | undefined): GradeSpec | undefined {
		if (!value) return undefined;
		return GRADE_SPECS.find((g) => g.value === value);
	}

	// Lookup a spec by its integer RANK (1=garbage.. 5=excellent). Returns
	// undefined for 0/out-of-range (ungraded). Used where the wire carries the
	// raw int rank (the report endpoint) rather than the boundary string.
	export function gradeSpecByRank(rank: number): GradeSpec | undefined {
		// GRADE_SPECS is best→worst (index 0 = excellent = rank 5), so rank r maps
		// to index (5 - r).
		if (rank < 1 || rank > GRADE_SPECS.length) return undefined;
		return GRADE_SPECS[GRADE_SPECS.length - rank];
	}

	// Re-export so callers (RunCard, +page) can render the same ungraded glyph.
	export { CircleDashedIcon };
</script>

<script lang="ts">
	import { api } from '$lib/api';

	// GradePicker shows a run's effective grade as a clickable smiley (or a dashed
	// circle when ungraded) and opens a dropdown to set/clear it. The effective
	// grade is the human `grade` if set, otherwise the cached critic
	// `report_grade`. A human grade renders with phosphor `fill` weight; a critic
	// fallback renders with `duotone` weight so the two are visually distinct.
	//
	// The dropdown lists the five grades (best→worst) with icon + label, marks
	// the critic's grade row with a subtle "· critic" tag, and ends with an unset
	// row that either re-defers to the critic grade ("Use critic grade") or clears
	// outright ("Clear grade") depending on whether a critic grade exists.

	type GradeRun = {
		run_id: string;
		grade: string | null;
		report_grade: string | null;
	};

	let { run, onmutated }: { run: GradeRun; onmutated?: () => void } = $props();

	// The human grade wins; the critic grade is the fallback shown otherwise.
	const fromCritic = $derived(!run.grade && !!run.report_grade);
	const effective = $derived(run.grade ?? run.report_grade);
	const spec = $derived(gradeSpec(effective));
	// Human-set → fill; critic-fallback → duotone; the dashed-circle ungraded
	// state uses regular weight via its own glyph.
	const weight = $derived<'fill' | 'duotone'>(fromCritic ? 'duotone' : 'fill');
	const buttonTitle = $derived(
		spec ? `${spec.label}${fromCritic ? ' (critic)' : ''}` : 'Set grade'
	);

	let open = $state(false);
	function toggle(e: MouseEvent) {
		e.stopPropagation();
		open = !open;
	}
	function close() {
		open = false;
	}
	// Same click-outside mechanic as RunCard's overflow menu: a document-level
	// listener closes the popover when the click lands outside the wrapper.
	function clickOutside(node: HTMLElement, callback: () => void) {
		const handle = (e: MouseEvent) => {
			if (!node.contains(e.target as Node)) callback();
		};
		document.addEventListener('click', handle);
		return { destroy: () => document.removeEventListener('click', handle) };
	}

	// Setting a grade sends the boundary string; clearing sends null so the
	// server resets to ungraded (then the critic fallback takes over again).
	async function pick(value: string | null, e: MouseEvent) {
		e.stopPropagation();
		close();
		await api.updateRun(run.run_id, { grade: value });
		onmutated?.();
	}
</script>

<div class="grade-wrap" use:clickOutside={close}>
	<button
		class="grade-btn"
		class:ungraded={!spec}
		style={spec ? `color:${spec.color}` : ''}
		title={buttonTitle}
		aria-label={buttonTitle}
		onclick={toggle}
	>
		{#if spec}
			{@const Icon = spec.icon}
			<Icon size={18} {weight} />
		{:else}
			<CircleDashedIcon size={18} weight="regular" />
		{/if}
	</button>
	{#if open}
		<div class="menu" role="menu">
			{#each GRADE_SPECS as g (g.value)}
				{@const Icon = g.icon}
				<button
					class="item"
					class:active={effective === g.value}
					onclick={(e) => pick(g.value, e)}
				>
					<span class="item-icon" style={`color:${g.color}`}>
						<Icon size={16} weight="fill" />
					</span>
					<span class="item-label">{g.label}</span>
					{#if run.report_grade === g.value}
						<span class="critic-tag" title="The keen-critic's grade">· critic</span>
					{/if}
				</button>
			{/each}
			<div class="divider" role="separator"></div>
			<!-- Unset row: with a critic grade present, clearing re-defers to it
			     ("Use critic grade"); otherwise it just clears outright. Both send
			     grade:null — the difference is only the label/intent. -->
			<button
				class="item unset"
				class:active={!run.grade}
				onclick={(e) => pick(null, e)}
			>
				<span class="item-icon"><CircleDashedIcon size={16} weight="regular" /></span>
				<span class="item-label">{run.report_grade ? 'Use critic grade' : 'Clear grade'}</span>
			</button>
		</div>
	{/if}
</div>

<style>
	.grade-wrap {
		position: relative;
		flex-shrink: 0;
	}
	/* The trigger mirrors RunCard's .iconbtn sizing so it sits flush where the
	   old star button used to live. Color comes from the rank (inline style);
	   the ungraded state falls back to a dim dashed circle. */
	.grade-btn {
		display: inline-flex;
		align-items: center;
		background: none;
		border: none;
		color: var(--text-dim);
		cursor: pointer;
		padding: 0.2rem;
		line-height: 1;
	}
	.grade-btn.ungraded {
		color: var(--text-dim);
		opacity: 0.75;
	}
	.grade-btn.ungraded:hover {
		opacity: 1;
		color: var(--text);
	}
	/* Popover styling matches RunCard's overflow menu (same vars) so the two
	   dropdowns feel like one system. Right-anchored: the picker now sits at the
	   row's trailing end (next to the ⋮ menu), so the menu opens leftward to stay
	   on-screen. */
	.menu {
		position: absolute;
		right: 0;
		top: calc(100% + 0.3rem);
		background: var(--bg-elev);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 0.25rem;
		z-index: 20;
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
		display: flex;
		flex-direction: column;
		min-width: 11rem;
	}
	.menu .item {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.4rem 0.6rem;
		background: none;
		border: none;
		color: var(--text);
		text-align: left;
		cursor: pointer;
		border-radius: calc(var(--radius) - 2px);
		font: inherit;
	}
	.menu .item:hover {
		background: var(--bg-elev2);
	}
	/* Currently-effective grade row gets a quiet gold marker (left border +
	   accented label) to show "this is what's set". */
	.menu .item.active {
		background: var(--bg-elev2);
	}
	.menu .item.active .item-label {
		color: var(--accent);
	}
	.item-icon {
		display: inline-flex;
		align-items: center;
		flex-shrink: 0;
	}
	.item-label {
		flex: 1;
	}
	/* "· critic" marker on the row matching the cached critic grade — subtle, so
	   it hints at the default without shouting. */
	.critic-tag {
		font-size: var(--fs-xs);
		color: var(--text-dim);
		font-style: italic;
	}
	.menu .unset {
		color: var(--text-dim);
	}
	.menu .divider {
		height: 1px;
		background: var(--border);
		margin: 0.25rem 0.1rem;
	}
</style>
