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
// (CiderIcon, NameWorkspaceModal, GcertChip); this OSS replacement exports a
// single no-op `Noop` component under each of those names plus
// `INTERNAL = false` so importers can statically gate corp-only rendering.
//
// The `as any` casts are deliberate. Each real component has a distinct prop
// interface (e.g. NameWorkspaceModal has a $bindable `open`) that a single
// Noop can't statically satisfy. Importers gate rendering on INTERNAL or
// field presence (see TopBanner.svelte / RunCard.svelte) so Noop never
// actually executes in OSS; the cast just unblocks svelte-check.

import Noop from './Noop.svelte';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const CiderIcon = Noop as any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const NameWorkspaceModal = Noop as any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const GcertChip = Noop as any;

export const INTERNAL = false;
