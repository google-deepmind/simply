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
import TerminalIcon from 'phosphor-svelte/lib/Terminal';
import FileTextIcon from 'phosphor-svelte/lib/FileText';
import PencilSimpleIcon from 'phosphor-svelte/lib/PencilSimple';
import MagnifyingGlassIcon from 'phosphor-svelte/lib/MagnifyingGlass';
import BrainIcon from 'phosphor-svelte/lib/Brain';
import AndroidLogoIcon from 'phosphor-svelte/lib/AndroidLogo';
import PaperPlaneTiltIcon from 'phosphor-svelte/lib/PaperPlaneTilt';
import HourglassMediumIcon from 'phosphor-svelte/lib/HourglassMedium';
import ListBulletsIcon from 'phosphor-svelte/lib/ListBullets';
import NotepadIcon from 'phosphor-svelte/lib/Notepad';
import NotebookIcon from 'phosphor-svelte/lib/Notebook';
import WrenchIcon from 'phosphor-svelte/lib/Wrench';

export type ToolIcon = Component<IconComponentProps>;

// Per-tool icon, shared by the chat tool pills and the <ToolCall> detail/
// trajectory renderer so a tool reads identically everywhere. All bash verbs
// share the terminal icon so a shell command reads distinctly from an internal
// tool of the same verb. Unknown tools fall back to the wrench.
const TOOL_ICONS: Record<string, ToolIcon> = {
	bash: TerminalIcon,
	view_file: FileTextIcon,
	edit_file: PencilSimpleIcon,
	recall_search: MagnifyingGlassIcon,
	recall_load: BrainIcon,
	spawn_agent: AndroidLogoIcon,
	send_message: PaperPlaneTiltIcon,
	await_event: HourglassMediumIcon,
	session_list: ListBulletsIcon,
	session_summary: NotepadIcon,
	session_search: MagnifyingGlassIcon,
	view_run_report: NotebookIcon
};

export const toolIcon = (name: string): ToolIcon => TOOL_ICONS[name] ?? WrenchIcon;
