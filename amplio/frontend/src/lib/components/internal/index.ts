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

// OSS stub barrel. The internal build provides the real corp components
// (editor icon, workspace-name modal, credential chip); this OSS replacement exports a
// single no-op `Noop` component under each of those names plus
// `INTERNAL = false` so importers can statically gate corp-only rendering.
//
// The `as any` casts are deliberate. Each real component has a distinct prop
// interface (e.g. the workspace-name modal has a $bindable `open`) that a single
// Noop can't statically satisfy. Importers gate rendering on INTERNAL or
// field presence (see TopBanner.svelte / RunCard.svelte) so Noop never
// actually executes in OSS; the cast just unblocks svelte-check.

import Noop from './Noop.svelte';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const EditorIcon = Noop as any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const WorkspaceNameModal = Noop as any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const GcertChip = Noop as any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const ExtraWorkspacePicker = Noop as any;

// No extra workspace sources in this build: the host's own path input governs.
export interface ExtraSelection {
	active: boolean;
	spec: string;
	summary: string;
	title: string;
	icon: null;
}
export const NO_SELECTION: ExtraSelection = {
	active: false,
	spec: '',
	summary: '',
	title: '',
	icon: null,
};
// Signature MUST match the corp implementation in
// frontend/src/lib/components/internal/workspaceModes.ts: the caller
// (WorkspaceField.svelte) is shared source and passes both arguments. The
// parameters are unused here — there are no extra workspace sources in this
// build — but omitting them only fails in the OSS build, which is the one place
// nobody runs by habit.
export function resolveExtraWorkspace(
	_state: Record<string, unknown>,
	_prefer: boolean,
): ExtraSelection {
	return NO_SELECTION;
}

export const INTERNAL = false;
