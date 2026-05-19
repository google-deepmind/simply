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
"""HTML Visualizer for the Agent memory."""

import html
import json
import re

from etils import epath

from simply.agent import memory as memory_lib


_BASH_COMMANDS = [
    'python',
    'python3',
    'pip',
    'head',
    'cat',
    'grep',
    'tail',
    'sed',
    'echo',
    'ls',
    'cd',
    'hg',
    'git',
    'date',
    'mkdir',
    'rm',
    'export',
    'wc',
    'wait',
    'find',
    'diff',
    'sleep',
    'pytest',
    'make',
    'bash',
]
_BASH_CMD_REGEX = re.compile(
    r'\b(' + '|'.join(re.escape(c) for c in _BASH_COMMANDS) + r')\b'
)


_CSS = """
body {
  font-family: sans-serif;
  margin: 20px auto;
  max-width: 1440px;
  background-color: #f4f4f4;
  color: #333;
}

h1, h2, h3 {
  color: #222;
}

.step-block {
  background: white;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 15px;
  margin-bottom: 20px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.step-summary {
  font-size: 1.2em;
  font-weight: bold;
  cursor: pointer;
  outline: none;
}

details > summary, .custom-summary {
  padding: 5px;
  background-color: #f1f1f1;
  border-radius: 4px;
  cursor: pointer;
  user-select: none;
}

details > summary:hover, .custom-summary:hover {
  background-color: #e2e2e2;
}

.event-block {
  margin-top: 10px;
  border-left: 3px solid #007bff;
  padding-left: 10px;
}

.event-block.llm-output {
  border-left-color: #28a745;
}

.event-summary {
  cursor: pointer;
}

.event-content {
  background: #f8f9fa;
  padding: 10px;
  padding-right: 45px;
  border-radius: 4px;
  overflow-x: auto;
  font-family: monospace;
  white-space: pre-wrap;
}


.file-block {
  margin-bottom: 5px;
}

.file-summary {
  font-family: monospace;
  background: #eef;
  cursor: pointer;
  padding: 2px 5px;
  border-radius: 3px;
}

.file-content {
  background: #f8f8f8;
  padding: 10px;
  font-family: monospace;
  white-space: pre-wrap;
  border: 1px solid #ddd;
  border-radius: 4px;
  max-height: 400px;
  overflow-y: auto;
}

.md-view {
  background: white;
  padding: 10px;
  padding-right: 45px;
  border-radius: 4px;
  border: 1px solid #ddd;
}

.md-view table {
  border-collapse: collapse;
}

.md-view th, .md-view td {
  border: 1px solid #ccc;
  padding: 4px 8px;
}

.md-view pre {
  background: #f6f8fa;
  padding: 12px;
  border-radius: 4px;
  overflow-x: auto;
}

.md-view code {
  background: #f0f0f0;
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 0.9em;
}

.md-view pre code {
  background: none;
  padding: 0;
}

.toggle-container {
  position: relative;
}

.toggle-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  cursor: pointer;
  padding: 4px 8px;
  background-color: #eef;
  border: 1px solid #ccc;
  border-radius: 4px;
  z-index: 10;
  opacity: 0.7;
  font-size: 1.2em;
  line-height: 1;
}

.toggle-btn:hover {
  opacity: 1;
}
"""

_JS = """
function toggleView(btn) {
  var container = btn.parentElement;
  var mdView = container.querySelector('.md-view');
  var rawView = container.querySelector('.raw-view');
  if (mdView.style.display === 'none') {
    mdView.style.display = 'block';
    rawView.style.display = 'none';
    btn.title = 'Show Raw Text';
  } else {
    mdView.style.display = 'none';
    rawView.style.display = 'block';
    btn.title = 'Show Markdown';
  }
}

// Render all markdown views on page load using Marked.js
document.addEventListener('DOMContentLoaded', function() {
  marked.setOptions({
    breaks: false,
    gfm: true,
  });
  document.querySelectorAll('.md-view[data-raw]').forEach(function(el) {
    el.innerHTML = marked.parse(el.getAttribute('data-raw'));
  });
});
"""


def format_memory_file(
    file: memory_lib.MemoryFile,
    display_override: memory_lib.DisplayMode | None = None,
) -> str:
  """Formats a MemoryFile as an interactive HTML details block."""
  display = display_override or file.display
  badge = f' [LLM saw: {display.value}]'
  open_attr = ' open' if display == memory_lib.DisplayMode.FULL else ''

  html_out = [
      f'<details class="file-block"{open_attr}>',
      (
          '  <summary'
          f' class="file-summary"><b>{html.escape(file.uri)}{html.escape(badge)}</b>'
          f' - <i>{html.escape(file.summary)}</i></summary>'
      ),
  ]
  if file.content:
    html_out.append(
        f'  <div class="file-content">{html.escape(file.content)}</div>'
    )
  html_out.append('</details>')
  return '\n'.join(html_out)


def format_event(file: memory_lib.MemoryFile) -> str:
  """Formats a log event for the holistic view."""
  event_type = file.metadata.get('event_type', 'unknown')
  css_class = (
      'event-block llm-output' if event_type == 'llm_output' else 'event-block'
  )

  title = (
      'Tool Call: ' + file.metadata.get('tool_name', 'Unknown')
      if event_type == 'tool_call'
      else 'LLM Output'
  )
  if event_type == 'summary':
    title = 'Compressed Summary'

  if event_type == 'llm_output' and file.content:
    escaped_raw = html.escape(file.content)
    content_html = f"""
    <div class="toggle-container">
      <button class="toggle-btn" onclick="toggleView(this)" title="Show Raw Text">📝</button>
      <div class="md-view" data-raw="{escaped_raw}"></div>
      <div class="raw-view" style="display: none;"><pre class="event-content" style="margin: 0;">{escaped_raw}</pre></div>
    </div>
    """
  else:
    content_html = (
        f'<div class="event-content">{html.escape(file.content)}</div>'
    )

  return f"""
  <details class="{css_class}">
    <summary class="event-summary"><b>{html.escape(title)}</b> - <i>{html.escape(file.summary)}</i></summary>
    {content_html}
  </details>
  """


def generate_token_usage_plot(
    snapshots: list[memory_lib.MemorySnapshot],
) -> str:
  """Generates a Chart.js plot for token usage against steps."""
  if not snapshots:
    return ''

  steps = [s.system_status.status_step for s in snapshots]
  usages = [s.system_status.approximate_token_usage for s in snapshots]
  budgets = [s.system_status.max_token_budget for s in snapshots]

  canvas_id = 'tokenUsageChart'

  html_out = [
      (
          '<div style="position: relative; height: 300px; width:'
          f' 100%;"><canvas id="{canvas_id}"></canvas></div>'
      ),
      '<script>',
      'document.addEventListener("DOMContentLoaded", function() {',
      f'  var ctx = document.getElementById("{canvas_id}").getContext("2d");',
      '  new Chart(ctx, {',
      '    type: "line",',
      '    data: {',
      f'      labels: {json.dumps(steps)},',
      '      datasets: [{',
      '        label: "Token Usage",',
      f'        data: {json.dumps(usages)},',
      '        borderColor: "#007bff",',
      '        backgroundColor: "#007bff",',
      '        borderWidth: 2,',
      '        pointRadius: 3,',
      '        fill: false,',
      '        tension: 0.1',
      '      }, {',
      '        label: "Max Budget",',
      f'        data: {json.dumps(budgets)},',
      '        borderColor: "#dc3545",',
      '        borderWidth: 2,',
      '        borderDash: [5, 5],',
      '        pointRadius: 0,',
      '        fill: false,',
      '        stepped: "middle"',
      '      }]',
      '    },',
      '    options: {',
      '      responsive: true,',
      '      maintainAspectRatio: false,',
      '      scales: {',
      '        y: {',
      '          beginAtZero: true',
      '        }',
      '      },',
      '      animation: false',
      '    }',
      '  });',
      '});',
      '</script>',
  ]
  return '\n'.join(html_out)


def generate_progress_section(
    progress_log: list[memory_lib.ProgressEntry],
) -> str:
  """Generates an HTML section with a Chart.js plot and table for progress."""
  if not progress_log:
    return ''

  # Group metric entries by metric name, preserving insertion order.
  # Each group collects (step, variant, value) triples.
  groups: dict[str, list[tuple[int, str, float]]] = {}
  for entry in progress_log:
    for m in entry.metrics:
      groups.setdefault(m.metric, []).append((entry.step, m.variant, m.value))

  # Build one scatter chart per metric name, with min/max envelope.
  chart_html: list[str] = []
  chart_idx = 0
  dot_color = '#007bff'
  envelope_color = 'rgba(0, 123, 255, 0.12)'

  for group_name, triples in groups.items():
    # Flatten all (step, value, tag) triples for scatter points.
    scatter_points: list[dict] = []  # pylint: disable=g-bare-generic
    # Collect values per step for envelope computation.
    values_by_step: dict[int, list[float]] = {}
    for step, variant, val in triples:
      scatter_points.append({'x': step, 'y': val, 'tag': variant})
      values_by_step.setdefault(step, []).append(val)

    # Compute running min/max envelope across steps seen so far.
    sorted_steps = sorted(values_by_step.keys())
    running_min = float('inf')
    running_max = float('-inf')
    envelope_min: list[dict] = []  # pylint: disable=g-bare-generic
    envelope_max: list[dict] = []  # pylint: disable=g-bare-generic
    for s in sorted_steps:
      for v in values_by_step[s]:
        running_min = min(running_min, v)
        running_max = max(running_max, v)
      envelope_min.append({'x': s, 'y': running_min})
      envelope_max.append({'x': s, 'y': running_max})

    canvas_id = f'progressChart_{chart_idx}'
    chart_idx += 1

    chart_html.extend([
        (
            '<div style="position: relative; height: 250px; width: 100%;'
            ' margin-bottom: 10px;">'
        ),
        f'  <canvas id="{canvas_id}"></canvas>',
        '</div>',
        '<script>',
        'document.addEventListener("DOMContentLoaded", function() {',
        f'  var ctx = document.getElementById("{canvas_id}").getContext("2d");',
        '  new Chart(ctx, {',
        '    type: "scatter",',
        '    data: {',
        '      datasets: [',
        # Scatter points dataset.
        '        {',
        '          label: "experiments",',
        f'          data: {json.dumps(scatter_points)},',
        f'          backgroundColor: "{dot_color}",',
        '          pointRadius: 5,',
        '          pointHoverRadius: 7',
        '        },',
        # Envelope max (line, filled down to min).
        '        {',
        '          label: "envelope max",',
        f'          data: {json.dumps(envelope_max)},',
        '          type: "line",',
        '          borderWidth: 0,',
        '          pointRadius: 0,',
        f'          backgroundColor: "{envelope_color}",',
        '          fill: "+1",',
        '          tension: 0.1',
        '        },',
        # Envelope min (line, no fill).
        '        {',
        '          label: "envelope min",',
        f'          data: {json.dumps(envelope_min)},',
        '          type: "line",',
        '          borderWidth: 0,',
        '          pointRadius: 0,',
        '          backgroundColor: "transparent",',
        '          fill: false,',
        '          tension: 0.1',
        '        }',
        '      ]',
        '    },',
        '    options: {',
        '      responsive: true,',
        '      maintainAspectRatio: false,',
        '      plugins: {',
        f'        title: {{ display: true, text: {json.dumps(group_name)} }},',
        '        legend: { display: false },',
        '        tooltip: {',
        '          callbacks: {',
        '            label: function(context) {',
        '              var p = context.raw;',
        '              if (p.tag) return p.tag + ": " + p.y;',
        '              return "";',
        '            }',
        '          }',
        '        }',
        '      },',
        '      scales: {',
        '        x: { title: { display: true, text: "Step" } }',
        '      },',
        '      animation: false',
        '    }',
        '  });',
        '});',
        '</script>',
    ])

  # Build table — one row per entry showing step, description, and metrics.
  table_html = [
      (
          '<table style="width: 100%; border-collapse: collapse; margin-top:'
          ' 15px;">'
      ),
      '  <thead><tr>',
      (
          '    <th style="padding: 8px; border: 1px solid #ccc; text-align:'
          ' left;">Step</th>'
      ),
      (
          '    <th style="padding: 8px; border: 1px solid #ccc; text-align:'
          ' left;">Description</th>'
      ),
      (
          '    <th style="padding: 8px; border: 1px solid #ccc; text-align:'
          ' left;">Metrics</th>'
      ),
      '  </tr></thead>',
      '  <tbody>',
  ]
  for entry in progress_log:
    metrics_str = ', '.join(
        f'{html.escape(m.variant)}/{html.escape(m.metric)}: {m.value:.4g}'
        for m in entry.metrics
    )
    table_html.append('    <tr>')
    table_html.append(
        '      <td style="padding: 8px; border: 1px solid'
        f' #ccc;">{entry.step}</td>'
    )
    table_html.append(
        '      <td style="padding: 8px; border: 1px solid'
        f' #ccc;">{html.escape(entry.description)}</td>'
    )
    table_html.append(
        '      <td style="padding: 8px; border: 1px solid #ccc;'
        f' font-family: monospace;">{metrics_str}</td>'
    )
    table_html.append('    </tr>')
  table_html.append('  </tbody>')
  table_html.append('</table>')

  parts = [
      (
          '<div class="step-block" style="background-color: #f0fff0;'
          ' border-color: #a3d9a5;">'
      ),
      '  <details open>',
      '    <summary class="step-summary"><b>Progress</b></summary>',
      '\n'.join(chart_html),
      '\n'.join(table_html),
      '  </details>',
      '</div>',
  ]
  return '\n'.join(parts)


def generate_html(
    mem_system: memory_lib.MemorySystem,
    out_path: epath.Path,
    agent_info: dict[str, str] | None = None,
    trajectory_critique: str | None = None,
    progress_log: list[memory_lib.ProgressEntry] | None = None,
):
  """Generates a holistic HTML overview of the conversation."""
  snapshots = mem_system.memory_snapshots
  memory_system_description = mem_system.memory_system_description

  html_parts = [
      '<!DOCTYPE html>',
      '<html lang="en">',
      '<head>',
      '<meta charset="utf-8">',
      '<meta name="viewport" content="width=device-width, initial-scale=1">',
      '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>',
      '<script',
      ' src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>',
      f'<style>{_CSS}</style>',
      f'<script>{_JS}</script>',
      '<title>Agent Conversation Visualizer</title>',
      '</head>',
      '<body>',
      '<h1>Agent Conversation Visualizer</h1>',
  ]

  if agent_info:
    html_parts.extend([
        '<div class="step-block" style="background-color: #f8f9fa;">',
        '  <h2>Agent Information</h2>',
        (
            '  <table style="width: 100%; max-width: 800px; border-collapse:'
            ' collapse;">'
        ),
    ])
    for k, v in agent_info.items():
      html_parts.extend([
          '    <tr>',
          (
              '      <th style="text-align: left; padding: 8px; border: 1px'
              f' solid #ccc; width: 30%;">{html.escape(str(k))}</th>'
          ),
          (
              '      <td style="padding: 8px; border: 1px solid'
              f' #ccc;">{html.escape(str(v))}</td>'
          ),
          '    </tr>',
      ])
    html_parts.extend([
        '  </table>',
        '</div>',
    ])

  html_parts.extend([
      (
          '<div class="step-block" style="background-color: #eef9ff;'
          ' border-color: #c4e1ff;">'
      ),
      '  <details style="margin-bottom: 10px">',
      '    <summary class="step-summary"><b>System Prompt</b></summary>',
      (
          '    <div'
          f' class="file-content">{html.escape(mem_system.system_prompt)}</div>'
      ),
      '  </details>',
      '  <details style="margin-bottom: 10px" open>',
      '    <summary class="step-summary"><b>Agent Task</b></summary>',
      f'    <div class="file-content">{html.escape(mem_system.task)}</div>',
      '  </details>',
      '  <details>',
      '    <summary class="step-summary"><b>System Description</b></summary>',
      (
          '    <div'
          f' class="file-content">{html.escape(memory_system_description)}</div>'
      ),
      '  </details>',
      '</div>',
  ])

  if progress_log:
    html_parts.append(
        generate_progress_section(progress_log)
    )

  if trajectory_critique:
    escaped_critique = html.escape(trajectory_critique)
    html_parts.extend([
        (
            '<div class="step-block" style="background-color: #fff8e6;'
            ' border-color: #ffd966;">'
        ),
        '  <details open>',
        (
            '    <summary class="step-summary"><b>Trajectory'
            ' Critique</b></summary>'
        ),
        '    <div class="toggle-container">',
        '      <button class="toggle-btn" onclick="toggleView(this)"',
        ' title="Show Raw Text">📝</button>',
        f'      <div class="md-view" data-raw="{escaped_critique}"></div>',
        ('      <div class="raw-view" style="display: none;">'
         f'<pre class="event-content" style="margin: 0;">{escaped_critique}'
         '</pre></div>'),
        '    </div>',
        '  </details>',
        '</div>',
    ])

  # Group events by step and count tool calls
  tool_counts = {}
  bash_cmd_counts = {}
  events_by_step = {}
  for snapshot in snapshots:
    step_idx = snapshot.system_status.status_step
    step_events = mem_system.get_events_for_step(step_idx)
    events_by_step[step_idx] = step_events
    for file in step_events:
      if file.metadata.get('event_type') == 'tool_call':
        tool_name = file.metadata.get('tool_name', 'Unknown')
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        if tool_name == 'bash':
          for cmd in set(_BASH_CMD_REGEX.findall(file.summary or '')):
            bash_cmd_counts[cmd] = bash_cmd_counts.get(cmd, 0) + 1

  # Render each step
  for step_idx, snapshot in enumerate(snapshots):
    html_parts.append('<div class="step-block">')
    html_parts.append('  <details open>')
    html_parts.append(
        f'    <summary class="step-summary">Step {step_idx}</summary>'
    )

    # --- Section 1: Meta info bar ---
    status = snapshot.system_status
    elapsed_s = int(status.elapsed_seconds)
    hours, remainder = divmod(elapsed_s, 3600)
    mins, _ = divmod(remainder, 60)
    if hours > 0:
      elapsed_str = f'{hours}h {mins}m'
    else:
      elapsed_str = f'{mins}m'
    budget = status.max_token_budget
    usage = status.approximate_token_usage
    usage_pct = int(100 * usage / budget) if budget > 0 else 0

    ctx_rel_path = f'context_view/ctx_for_step_{step_idx:06d}.md'

    html_parts.append(
        '    <div style="margin-top: 8px; font-size: 0.85em; color: #888;">'
        f'⏱ {elapsed_str}'
        f' &nbsp;·&nbsp; 📊 tokens {usage_pct}%'
        f' ({usage}/{budget})'
        f' &nbsp;·&nbsp; <a href="{ctx_rel_path}" target="_blank"'
        ' style="color: #888; text-decoration: none;">'
        '🔍 context view</a>'
        '</div>'
    )

    # Separate LLM output events from other events
    events = events_by_step.get(step_idx, [])
    llm_events = [
        e for e in events
        if e.metadata.get('event_type') == 'llm_output'
    ]
    other_events = [
        e for e in events
        if e.metadata.get('event_type') != 'llm_output'
    ]

    # --- Section 2: LLM Output (always visible) ---
    if step_idx == 0:
      html_parts.append(
          '    <div style="margin-top: 15px;"><i>Initial State</i></div>'
      )
    elif llm_events:
      html_parts.append('    <div style="margin-top: 15px;">')
      for llm_file in llm_events:
        if llm_file.content:
          escaped_raw = html.escape(llm_file.content)
          html_parts.append(f"""
          <div class="toggle-container">
            <button class="toggle-btn" onclick="toggleView(this)" title="Show Raw Text">📝</button>
            <div class="md-view" data-raw="{escaped_raw}"></div>
            <div class="raw-view" style="display: none;"><pre class="event-content" style="margin: 0;">{escaped_raw}</pre></div>
          </div>""")
      html_parts.append('    </div>')

    # --- Section 3: Other events (tool calls, summaries, etc.) ---
    if other_events:
      html_parts.append('    <div style="margin-top: 10px;">')
      for event_file in other_events:
        html_parts.append(format_event(event_file))
      html_parts.append('    </div>')

    html_parts.append('  </details>')
    html_parts.append('</div>')

  if tool_counts:
    max_count = max(tool_counts.values())
    html_parts.append('<div class="step-block">')
    html_parts.append('  <h2>Tool Call Histogram</h2>')
    html_parts.append(
        '  <table style="width: 100%; max-width: 600px; border-collapse:'
        ' separate; border-spacing: 0 5px;">'
    )
    for name, count in sorted(
        tool_counts.items(), key=lambda x: x[1], reverse=True
    ):
      width_pct = max(1, int(100 * count / max_count))
      html_parts.append('    <tr>')
      html_parts.append(
          '      <td style="width: 250px; padding: 4px; font-family:'
          ' monospace; border: none; white-space:'
          f' nowrap;">{html.escape(name)}</td>'
      )
      html_parts.append(
          '      <td style="padding: 4px; border: none; width: 100%;"><div'
          ' style="display: flex; align-items: center;"><div'
          f' style="background-color: #007bff; width: {width_pct}%; height:'
          ' 20px; border-radius: 3px; margin-right: 10px;"></div><span'
          f' style="font-weight: bold;">{count}</span></div></td>'
      )
      html_parts.append('    </tr>')
      if name == 'bash' and bash_cmd_counts:
        for bash_cmd, b_count in sorted(
            bash_cmd_counts.items(), key=lambda x: x[1], reverse=True
        ):
          if b_count == 0:
            continue
          b_width_pct = max(1, int(100 * b_count / max_count))
          html_parts.append('    <tr>')
          html_parts.append(
              '      <td style="width: 250px; padding: 4px; padding-left:'
              ' 20px; font-family: monospace; border: none; color: #555;'
              f' white-space: nowrap;">‣ {html.escape(bash_cmd)}</td>'
          )
          html_parts.append(
              '      <td style="padding: 4px; border: none; width: 100%;"><div'
              ' style="display: flex; align-items: center;"><div'
              f' style="background-color: #6c757d; width: {b_width_pct}%;'
              ' height: 15px; border-radius: 3px; margin-right:'
              ' 10px;"></div><span style="font-weight: bold; color:'
              f' #555;">{b_count}</span></div></td>'
          )
          html_parts.append('    </tr>')
    html_parts.append('  </table>')
    html_parts.append('</div>')

  if snapshots:
    html_parts.append('<div class="step-block">')
    html_parts.append('  <h2>Token Usage over Steps</h2>')
    html_parts.append(
        f'  <div>\n{generate_token_usage_plot(snapshots)}\n  </div>'
    )
    html_parts.append('</div>')

  html_parts.append('</body></html>')

  out_path.write_text('\n'.join(html_parts))
