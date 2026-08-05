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
"""Main entry point for the Simply Agent."""

from collections.abc import Sequence
import datetime
import logging as python_logging
import os
import pickle
import uuid

from absl import app
from absl import flags
from absl import logging as absl_logging
from etils import epath
import frontmatter
from simply.agent import agent as agent_lib
from simply.agent import critique as critique_lib
from simply.agent import env as env_lib
from simply.agent import llm as llm_lib
from simply.agent import memory as memory_lib
from simply.agent import skill_loader as skill_loader_lib
from simply.agent import tui as tui_lib
from simply.agent import visualizer


_LLM = flags.DEFINE_string(
    'llm',
    'LiteLLM:vertex_ai/gemini-2.5-pro',
    'The LLM to use for the agent.',
)
_SYSTEM_LLM = flags.DEFINE_string(
    'system_llm',
    'LiteLLM:vertex_ai/gemini-2.5-pro',
    'The LLM to use for system tasks such as trajectory critique.',
)
_RUN_ID = flags.DEFINE_string(
    'run_id',
    None,
    'A unique ID for the run, randomly generated if not set.',
)
_TASK_FILE = flags.DEFINE_string(
    'task_file',
    None,
    'A markdown file containing the task description. If it is a directory, '
    'then in that directory, `task.md` describes the task, and '
    '`knowledge/*.md` files are predefined knowledge useful for the task.',
    required=True,
)

_DEFAULT_ENV = 'Local:.'
_ENV = flags.DEFINE_string(
    'env',
    _DEFAULT_ENV,
    'The execution environment spec in "provider:param" format.'
)
_RESUME = flags.DEFINE_bool(
    'resume',
    False,
    'Resume from a previously saved run. Requires --run_id.',
)
_BASE_SYSTEM_DIR = flags.DEFINE_string(
    'base_system_dir',
    '~/.simply_agent',
    'The base system directory.',
)
_MAX_NUM_SKILLS = flags.DEFINE_integer(
    'max_num_skills',
    32,
    'The max number of skills to load.',
)
_DEFAULT_SKILLS_DIR = '~/.simply_agent/skills'
_SKILLS_DIR = flags.DEFINE_string(
    'skills_dir',
    _DEFAULT_SKILLS_DIR,
    'Directory containing skill definitions.',
)

_DISPLAY_FULL = flags.DEFINE_bool(
    'display_full',
    False,
    'Whether to display everything or just a concise status',
)


def setup_logging(session_dir: epath.Path):
  """Sets up logging for the agent."""
  python_logging.getLogger('LiteLLM').setLevel(python_logging.WARNING)

  log_dir = session_dir / 'logs'
  log_dir.mkdir(parents=True, exist_ok=True)

  log_file = log_dir / (
      datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.txt'
  )
  log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
  formatter = python_logging.Formatter(log_format)
  file_stream = log_file.open('w')
  handler = python_logging.StreamHandler(file_stream)
  handler.setLevel(python_logging.INFO)
  handler.setFormatter(formatter)
  absl_logging.get_absl_logger().addHandler(handler)


def load_task(task_file_path: epath.Path) -> str:
  """Loads a task description from a markdown file."""
  raw = task_file_path.read_text()
  post = frontmatter.loads(raw)
  return post.content


def parse_predefined_knowledge(
    knowledge_file_path: epath.Path,
) -> memory_lib.MemoryFile:
  """Parses the predefined knowledge."""
  knowledge = knowledge_file_path.read_text()
  post = frontmatter.loads(knowledge)
  name = post.get('name', knowledge_file_path.stem)
  content = post.content
  kwargs = {'uri': 'kb://' + name, 'content': content, 'update_step': 0}  # pyrefly: ignore[unsupported-operation]
  kwargs['summary'] = post.get('summary', '')
  kwargs['display'] = memory_lib.DisplayMode(
      post.get('display', memory_lib.DisplayMode.SUMMARY.value)
  )
  return memory_lib.MemoryFile(**kwargs)  # pyrefly: ignore[missing-argument]


def run_agent_loop(
    agent: agent_lib.Agent,
    critique: critique_lib.TrajectoryCritique,
    system_llm: llm_lib.LLMBase,
    run_info: dict[str, str],
):
  """Runs the agent loop, calling step() and saving after each step."""

  while True:
    final_output = agent.step()
    agent.save_memory_snapshot()

    step = agent.memory_system.last_snapshot.system_status.status_step
    events = agent.memory_system.get_events_for_step(step)
    critique.add_step(step, events, system_llm)
    if final_output is not None:
      critique.finalize(system_llm)
    agent_lib.save_pickle_atomic(
        critique, agent.session_dir / 'critique.pkl'
    )

    progress_log = agent.memory_system.progress_log

    visualizer.generate_html(
        agent.memory_system,
        agent.session_dir / 'index.html',
        agent_info=run_info,
        trajectory_critique=critique.critique_text,
        progress_log=progress_log,
    )
    if final_output is not None:
      agent.tui.update_status(f'\u2705 Final Output:\n{final_output}')
      break


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  resuming = _RESUME.value

  task_file = epath.Path(_TASK_FILE.value)
  run_id = _RUN_ID.value
  if resuming and not run_id:
    raise app.UsageError('--run_id is required when using --resume.')
  if not run_id:
    run_id = (
        task_file.stem
        + '_'
        + datetime.datetime.now().strftime('%y%m%d_')
        + uuid.uuid4().hex[:4]
    )

  base_system_dir = epath.Path(_BASE_SYSTEM_DIR.value).expanduser().resolve()
  session_dir = base_system_dir / 'sessions' / run_id

  if resuming:
    if not session_dir.exists():
      raise FileNotFoundError(
          f'Session directory {session_dir} does not exist. Cannot resume.'
      )
  else:
    if session_dir.exists():
      raise FileExistsError(
          f'Session directory {session_dir} already exists, please use a'
          ' unique run_id.'
      )
    session_dir.mkdir(parents=True, exist_ok=True)

  setup_logging(session_dir)

  absl_logging.info('Simply Agent starting...')
  absl_logging.info('Session directory: %s', session_dir)
  absl_logging.info('Run ID: %s', run_id)

  system_llm = llm_lib.LLMRegistry.get_llm(_SYSTEM_LLM.value)

  if _DISPLAY_FULL.value:
    tui_display = tui_lib.FullDisplay()
  else:
    tui_display = tui_lib.StatusDisplay()

  if resuming:
    # -- resuming from a previous session --
    pkl_path = session_dir / 'memory_system.pkl'
    if not pkl_path.exists():
      raise FileNotFoundError(
          f'Memory system pickle not found at {pkl_path}. Cannot resume.'
      )
    absl_logging.info('Loading memory system from %s...', pkl_path)
    with pkl_path.open('rb') as f:
      loaded_memory_system = pickle.load(f)

    critique_pkl_path = session_dir / 'critique.pkl'
    if critique_pkl_path.exists():
      absl_logging.info('Loading critique from %s...', critique_pkl_path)
      with critique_pkl_path.open('rb') as f:
        critique = pickle.load(f)
    else:
      critique = critique_lib.TrajectoryCritique(
          task=loaded_memory_system.task,
      )

    task = loaded_memory_system.task
    env = env_lib.EnvRegistry.get_env(_ENV.value)
    agent = agent_lib.Agent(
        task=task,
        env=env,
        predefined_knowledge=[],
        llm_scheme=_LLM.value,
        session_dir=session_dir,
        tui=tui_display,
    )
    agent.restore_memory_system(loaded_memory_system)

  else:
    # -- brand new session --
    if task_file.is_dir():
      task = load_task(task_file / 'task.md')
      predefined_knowledge = [
          parse_predefined_knowledge(f)
          for f in task_file.glob('knowledge/*.md')
      ]
    else:
      task = load_task(task_file)
      predefined_knowledge = []

    if _MAX_NUM_SKILLS.value > 0:
      skill_cache_dir = base_system_dir / 'skill_cache'
      skill_memory_files = skill_loader_lib.load_skills_cached(
          skills_dir=_SKILLS_DIR.value,
          task=task,
          system_llm=system_llm,
          max_skills=_MAX_NUM_SKILLS.value,
          cache_dir=skill_cache_dir,
      )
      predefined_knowledge.extend(skill_memory_files)

    env = env_lib.EnvRegistry.get_env(_ENV.value)
    agent = agent_lib.Agent(
        task=task,
        env=env,
        predefined_knowledge=predefined_knowledge,
        llm_scheme=_LLM.value,
        session_dir=session_dir,
        tui=tui_display,
    )
    agent.save_memory_snapshot()
    critique = critique_lib.TrajectoryCritique(task=agent.task)

  run_info = {
      'LLM': _LLM.value,
      'System LLM': _SYSTEM_LLM.value,
      'Env': str(env),
      'Run ID': run_id,
  }
  tui_display.set_header(run_info)
  tui_display.set_task(agent.task)
  tui_display.start()
  try:
    run_agent_loop(agent, critique, system_llm, run_info)
  finally:
    tui_display.stop()

  absl_logging.info('Simply Agent finished.')


if __name__ == '__main__':
  app.run(main)
