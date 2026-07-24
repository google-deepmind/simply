/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import type { Component } from 'svelte';
import type { IconComponentProps } from 'phosphor-svelte';
import BirdIcon from 'phosphor-svelte/lib/Bird';
import BugIcon from 'phosphor-svelte/lib/Bug';
import ButterflyIcon from 'phosphor-svelte/lib/Butterfly';
import CatIcon from 'phosphor-svelte/lib/Cat';
import DogIcon from 'phosphor-svelte/lib/Dog';
import FishIcon from 'phosphor-svelte/lib/Fish';
import HorseIcon from 'phosphor-svelte/lib/Horse';
import RabbitIcon from 'phosphor-svelte/lib/Rabbit';
import SealIcon from 'phosphor-svelte/lib/Seal';
import ShrimpIcon from 'phosphor-svelte/lib/Shrimp';

// Maps the noun portion of a session/run nickname (adj-noun) to a
// best-fit phosphor animal icon. Coverage is ~58% of the noun corpus;
// unmapped nouns simply render without an animal icon (the existing
// agent-type identity icon still conveys chatbot-vs-autonomous).
//
// Phosphor lacks dedicated icons for many animals; we collapse families
// into their nearest visual cousin (e.g. all small birds → Bird, all
// cats → Cat). `Mouse` and `Crane` in phosphor are computer mouse and
// construction crane respectively, NOT animals, so they're excluded.
export type AnimalIcon = Component<IconComponentProps>;

const ICON_BY_NOUN: Record<string, AnimalIcon> = {
	// Birds — every bird in the corpus collapses to the generic Bird silhouette.
	bird: BirdIcon,
	condor: BirdIcon,
	crane: BirdIcon, // the bird sense; phosphor's Crane is a construction crane
	crow: BirdIcon,
	dove: BirdIcon,
	duck: BirdIcon,
	eagle: BirdIcon,
	egret: BirdIcon,
	emu: BirdIcon,
	falcon: BirdIcon,
	finch: BirdIcon,
	goose: BirdIcon,
	grouse: BirdIcon,
	gull: BirdIcon,
	hawk: BirdIcon,
	hen: BirdIcon,
	heron: BirdIcon,
	ibis: BirdIcon,
	jay: BirdIcon,
	kite: BirdIcon,
	lark: BirdIcon,
	owl: BirdIcon,
	parrot: BirdIcon,
	pelican: BirdIcon,
	penguin: BirdIcon,
	pigeon: BirdIcon,
	puffin: BirdIcon,
	quail: BirdIcon,
	raven: BirdIcon,
	robin: BirdIcon,
	sparrow: BirdIcon,
	stork: BirdIcon,
	swallow: BirdIcon,
	swan: BirdIcon,
	tern: BirdIcon,
	thrush: BirdIcon,

	// Bugs
	ant: BugIcon,
	bee: BugIcon,
	moth: ButterflyIcon,

	// Cats
	cat: CatIcon,
	lion: CatIcon,
	lynx: CatIcon,
	tiger: CatIcon,

	// Canines — fox is a canid; coyote/dingo/husky/jackal all map to Dog.
	coyote: DogIcon,
	dingo: DogIcon,
	fox: DogIcon,
	husky: DogIcon,
	jackal: DogIcon,

	// Fish
	fish: FishIcon,
	salmon: FishIcon,
	trout: FishIcon,

	// Equines
	horse: HorseIcon,
	pony: HorseIcon,

	// Lagomorphs
	bunny: RabbitIcon,
	hare: RabbitIcon,
	rabbit: RabbitIcon,

	// Marine
	seal: SealIcon,
	crab: ShrimpIcon,
	squid: ShrimpIcon,
};

// Returns the animal icon for a nickname, or null if the noun has no
// mapping (or the name is a reserved non-animal like main-agent /
// chatty-bot). Pass a session_id, run_id, or any "adj-noun" string.
export function iconForName(name: string | undefined | null): AnimalIcon | null {
	if (!name) return null;
	const noun = name.split('-').pop()?.toLowerCase() ?? '';
	return ICON_BY_NOUN[noun] ?? null;
}
