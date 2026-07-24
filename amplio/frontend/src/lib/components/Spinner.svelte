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
	// Shared border-circle spinner for in-progress banners/buttons. Inherits
	// currentColor so it tints to its context. `size` is the diameter in px.
	let { size = 12 }: { size?: number } = $props();
</script>

<span
	class="spinner spin"
	aria-hidden="true"
	style="width:{size}px;height:{size}px"
></span>

<style>
	.spinner {
		display: inline-block;
		border: 2px solid color-mix(in srgb, currentColor 35%, transparent);
		border-top-color: currentColor;
		border-radius: 50%;
		flex-shrink: 0;
	}
</style>
