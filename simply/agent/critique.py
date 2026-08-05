# Copyright 2026 The Simply Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trajectory auto-critique for the agent."""

import dataclasses
from typing import Annotated

from absl import logging

from simply.agent import llm as llm_lib
from simply.agent import memory as memory_lib


_CRITIQUE_SYSTEM_PROMPT = r"""
You are an expert analytical engine designed to evaluate the execution
trajectories of autonomous AI agents. To avoid token overflow, you will analyze
the trajectory in segments. Each time, you will be given the task description,
any previous critiques for previous segments, and the events from the current
segment to critique. Your goal is to provide a concise critique covering the
highlights of the current segment.

Your goal is to distill the execution logs into a structured analysis that
captures the overarching narrative (strategy, success/failure) alongside
precise, granular artifacts (code, metrics, identifiers). You must format your
response strictly using the following sections:

**1. High-Level Segment Summary**
Provide a concise narrative of the agent's behavior within the current segment.
*   **Segment Goal:** What was the agent actively trying to accomplish?
*   **Status & Outcomes:** Did the agent succeed, fail, or pivot in this segment?
    Explicitly mention any dead ends, looping behaviors, or successful task
    completions.
*   **Segment statistics:** Number of steps, time spent, number of jobs
    launched, number of hg commits, etc.
*   **Step Groupings:** Summarize the sequence of actions into logical phases
    (e.g., "Explored repository -> Modified data loader -> Launched training").

**2. Group of Steps and Low-Level Artifacts & Details**
Partition the current segment into logical groups of steps. Then, for each
group, provide a concise description and list all critical artifacts:
*   **Files & Paths:** Specific filenames, directories, or code blocks
    created/modified.
*   **Identifiers:** Experiment IDs, job IDs, commit hashes, or URLs.
*   **Metrics & Results:** Intermediate or final accuracies, loss values,
    test results, or performance benchmarks.
*   **Errors/Exceptions:** Exact error messages or stack trace summaries that
    caused the agent to fail or pivot.
""".strip()

_CRITIQUE_PROMPT = r"""
# Task

{task}

# Previous critiques

{previous_critiques_section}

# Events in the current segment (Step {start_step}-{end_step})

{events_text}
""".strip()

_FINAL_SUMMARY_SYSTEM_PROMPT = r"""
You are an expert analytical engine. You have already produced segment-level
critiques for an autonomous AI agent's execution trajectory. Now you must
produce a single **executive summary** that synthesizes all segments into a
cohesive review.
""".strip()

_FINAL_SUMMARY_PROMPT = r"""
# Task

{task}

# Segment Critiques

{all_critiques}

# Instructions

Synthesize the above segment critiques into a concise **Executive Summary**.
Cover:
1. **Overall Strategy & Arc:** What was the agent's high-level approach? Did it
   evolve or pivot over time?
2. **Key Accomplishments:** The most significant milestones or results achieved.
3. **Critical Failures & Inefficiencies:** Major errors, dead-ends, or wasted
   effort.
4. **Final Outcome:** Did the agent ultimately succeed, partially succeed, or
   fail at the task?
5. **Important statistics:** Total number of steps, total time spent, number
   of jobs launched, number of hg commits, etc.
""".strip()


def _format_step_events(
    step: int, events: list[memory_lib.MemoryFile]
) -> str:
  """Formats a step's events into a concise text block for the critique LLM."""
  parts = [f'## Step {step}'] + [
      event.to_llm(memory_lib.DisplayMode.FULL) for event in events
  ]
  return '\n\n'.join(parts)


@dataclasses.dataclass
class SegmentCritique:
  """A critique summary produced by the system LLM for one segment."""
  start_step: int
  end_step: int
  summary: str


@dataclasses.dataclass
class TrajectoryCritique:
  """Auto-critiques the agent's trajectory in segments.

  This class eagerly formats events into prompt text and maintains a running
  token count. When the pending content exceeds the configured fraction of
  the system LLM's token budget, it triggers a summarization call.

  Call finalize() at the end of the agent loop to flush any pending events
  and generate an executive summary across all segments.
  """

  task: Annotated[str, 'The agent task description.']
  token_budget_fraction: Annotated[
      float,
      'Fraction of system LLM token budget that triggers summarization.',
  ] = 0.3

  # Completed segment critiques.
  segment_critiques: Annotated[
      list[SegmentCritique], 'Completed segment critiques.'
  ] = dataclasses.field(default_factory=list)

  # Executive summary across all segments (generated by finalize()).
  final_summary: Annotated[
      str | None, 'Executive summary across all segments.'
  ] = None

  # Pending (unsummarized) segment state — eagerly formatted for efficiency.
  _pending_parts: Annotated[
      list[str], 'Eagerly formatted text blocks for pending events.'
  ] = dataclasses.field(default_factory=list)
  _pending_start_step: int | None = None
  _pending_end_step: int | None = None
  _pending_token_count: int = 0

  def add_step(
      self, step: int, events: list[memory_lib.MemoryFile], llm: llm_lib.LLMBase
  ):
    """Adds events from a step to the pending buffer.

    Events are eagerly formatted into text and the running token count is
    updated, so that later checks in `maybe_summarize` are O(1).

    Args:
      step: The step index.
      events: The events from this step.
      llm: The system LLM to use for the critique.
    """
    if not events:
      return
    formatted = _format_step_events(step, events)
    self._pending_parts.append(formatted)
    self._pending_token_count += llm.count_tokens(
        [{'role': 'user', 'content': formatted}]
    )
    if self._pending_start_step is None:
      self._pending_start_step = step
    self._pending_end_step = step
    self._maybe_summarize(llm)

  def finalize(self, llm: llm_lib.LLMBase):
    """Finalize the critique: flush pending events and generate a final summary.

    Forces summarization of any remaining pending events, then — if there are
    multiple segment critiques — asks the LLM to produce an executive summary
    across all segments.  For single-segment runs the segment critique already
    captures everything, so no additional summary is generated.

    Args:
      llm: The system LLM to use for the critique.
    """
    self._maybe_summarize(llm, force=True)

    # Only generate a final summary when there are multiple segments.
    if len(self.segment_critiques) <= 1:
      return

    all_critiques = '\n\n\n'.join(
        f'## Segment step {seg.start_step}-{seg.end_step}\n\n{seg.summary}'
        for seg in self.segment_critiques
    )
    prompt = _FINAL_SUMMARY_PROMPT.format(
        task=self.task,
        all_critiques=all_critiques,
    )

    logging.info(
        'Generating final executive summary across %d segments...',
        len(self.segment_critiques),
    )

    try:
      output = llm.completion(
          system_prompt=_FINAL_SUMMARY_SYSTEM_PROMPT,
          messages=[{'role': 'user', 'content': prompt}],
          tools=[],
          num_retries=3,
      )
      self.final_summary = output.text.strip()
    except Exception as e:  # pylint: disable=broad-except
      logging.error('Final summary generation failed: %s', e)
      self.final_summary = f'(Final summary generation failed: {e})'

  def _maybe_summarize(
      self,
      llm: llm_lib.LLMBase,
      force: bool = False,
  ) -> bool:
    """Triggers a critique summarization if the pending buffer is large enough.

    Args:
      llm: The system LLM to use for the critique.
      force: If True, summarize even if the token budget is not exceeded.
        Useful at the end of the agent loop.

    Returns:
      True if a summarization was performed.
    """
    if not self._pending_parts:
      return False

    if not force:
      threshold = int(self.token_budget_fraction * llm.max_tokens)
      if self._pending_token_count < threshold:
        return False

    # Build the prompt
    parts = [
        f'## Segment step {seg.start_step}-{seg.end_step}\n\n{seg.summary}'
        for seg in self.segment_critiques
    ]
    previous_critiques_section = '\n\n\n'.join(parts)

    events_text = '\n\n'.join(self._pending_parts)
    prompt = _CRITIQUE_PROMPT.format(
        task=self.task,
        previous_critiques_section=previous_critiques_section,
        start_step=self._pending_start_step,
        end_step=self._pending_end_step,
        events_text=events_text,
    )

    logging.info(
        'Triggering trajectory critique for steps %d-%d '
        '(pending tokens: %d)...',
        self._pending_start_step,
        self._pending_end_step,
        self._pending_token_count,
    )

    try:
      output = llm.completion(
          system_prompt=_CRITIQUE_SYSTEM_PROMPT,
          messages=[{'role': 'user', 'content': prompt}],
          tools=[],
          num_retries=3,
      )
      summary = output.text.strip()
    except Exception as e:  # pylint: disable=broad-except
      logging.error('Trajectory critique failed: %s', e)
      summary = f'(Critique generation failed: {e})'

    self.segment_critiques.append(SegmentCritique(
        start_step=self._pending_start_step,  # pyrefly: ignore[bad-argument-type]
        end_step=self._pending_end_step,  # pyrefly: ignore[bad-argument-type]
        summary=summary,
    ))

    # Reset pending state
    self._pending_parts = []
    self._pending_start_step = None
    self._pending_end_step = None
    self._pending_token_count = 0

    return True

  @property
  def critique_text(self) -> str:
    """Critique text for all completed segments.

    If a final summary has been generated (via `finalize()`), it is prepended.
    Pending (unsummarized) events are NOT included.

    Returns:
      A formatted string with all segment critiques, or empty string if none.
    """
    if not self.segment_critiques:
      return ''
    parts = []
    if self.final_summary is not None:
      parts.append(self.final_summary)
    for seg in self.segment_critiques:
      parts.append(
          f'## Segment step {seg.start_step}-{seg.end_step}\n\n{seg.summary}'
      )
    return '\n\n\n'.join(parts)
