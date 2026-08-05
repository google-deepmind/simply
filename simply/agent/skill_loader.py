"""Load skills and filter them based on relevance to a given task.

We implement a simple skill loader that uses the system LLM to evaluate
all the skill relevance at the beginning of the task, and put the
relevant skills in the memory system.
"""

import dataclasses
import datetime
import hashlib
import json
from typing import Callable, Sequence

from absl import logging
from etils import epath
import frontmatter
import rich.console
import rich.progress
import rich.table

from simply.agent import llm as llm_lib
from simply.agent import memory as memory_lib


@dataclasses.dataclass(frozen=True)
class SkillInfo:
  """Information about a skill parsed from SKILL.md."""

  name: str
  description: str
  content: str
  folder_name: str

  def __str__(self) -> str:
    return (
        f'### Skill: {self.name}\n\n'
        + f'**Description:** {self.description}\n\n'
        + f'**Content:**\n{self.content}'
    )


@dataclasses.dataclass(frozen=True)
class RatedSkill:
  """A skill with a relevance rating from the system LLM."""

  skill: SkillInfo
  rating: int  # 0-5
  updated_description: str


def _load_skill_files(skills_dir: epath.Path) -> list[SkillInfo]:
  """Load all SKILL.md files from skill directories."""
  skills = []
  for entry in sorted(skills_dir.iterdir()):
    if (
        entry.name.startswith('.')
        or entry.name.startswith('_')
        or not entry.is_dir()
    ):
      continue
    skill_file = entry / 'SKILL.md'
    if not skill_file.exists():
      logging.warning('No SKILL.md found in %s, skipping.', entry)
      continue
    try:
      text = skill_file.read_text()
      post = frontmatter.loads(text)
      name = post.get('name', entry.name)
      description = post.get('description', '')
      content = post.content
      skills.append(
          SkillInfo(
              name=name,  # pyrefly: ignore[bad-argument-type]
              description=description,  # pyrefly: ignore[bad-argument-type]
              content=content,
              folder_name=entry.name,
          )
      )
    except Exception as e:  # pylint: disable=broad-except
      logging.warning('Failed to parse SKILL.md in %s: %s', entry, e)
  return skills


_RELEVANCE_SYSTEM_PROMPT = """
You are a skill relevance evaluator. You will be given a task description and
one or more skills. For each skill, you must determine whether it is potentially
relevant to the task.

For each skill, respond with a JSON object on a separate line with the following
fields:

- "skill_name": the name of the skill
- "rating": an integer from 0 to 5 indicating relevance (0 = completely
  irrelevant, 5 = highly relevant / essential)
- "description": a concise description of the skill that would help an agent
  decide whether to read the full skill content. Output empty string if the
  original description is already good, or improve it if needed.

Output ONLY the JSON objects, one per line, no other text. The order must match
the order of the skills presented.
""".strip()


def _build_relevance_prompt(
    task: str,
    skills: Sequence[SkillInfo],
) -> str:
  """Build the user prompt for relevance evaluation."""
  parts = [f'## Task\n\n{task}\n\n## Skills to evaluate\n']
  for skill in enumerate(skills):
    parts.append(str(skill))
  return '\n\n'.join(parts)


def _batch_skills(
    task: str,
    skills: list[SkillInfo],
    max_tokens_per_call: int,
    token_counter: Callable[[str], int],
) -> list[list[SkillInfo]]:
  """Split skills into batches that fit within the token budget."""
  task_tokens = token_counter(task)
  system_prompt_tokens = token_counter(_RELEVANCE_SYSTEM_PROMPT)
  overhead = task_tokens + system_prompt_tokens + 500  # buffer for formatting

  batches: list[list[SkillInfo]] = []
  current_batch: list[SkillInfo] = []
  current_tokens = overhead

  for skill in skills:
    skill_tokens = token_counter(str(skill))

    if current_batch and current_tokens + skill_tokens > max_tokens_per_call:
      batches.append(current_batch)
      current_batch = []
      current_tokens = overhead

    current_batch.append(skill)
    current_tokens += skill_tokens

  if current_batch:
    batches.append(current_batch)

  return batches


def _parse_llm_response(
    response_text: str,
    skills: Sequence[SkillInfo],
) -> list[tuple[int, str]]:
  """Parse LLM response into (rating, description) pairs.

  Returns a list aligned with the input skills. If parsing fails for
  some entries, uses defaults (rating=0, original description).

  Args:
    response_text: The LLM response text.
    skills: The list of skills that were passed to the LLM.

  Returns:
    A list of (rating, description) pairs, one per input skill.
  """
  results: list[tuple[int, str]] = []

  # Try to parse line by line
  lines = [
      line.strip() for line in response_text.strip().split('\n') if line.strip()
  ]

  # Filter to only JSON-looking lines
  json_lines = []
  for line in lines:
    # Strip markdown code fences if present
    cleaned = line.strip('`').strip()
    if cleaned.startswith('json'):
      cleaned = cleaned.removeprefix('json').strip()
    if cleaned.startswith('{'):
      json_lines.append(cleaned)

  for i, skill in enumerate(skills):
    if i < len(json_lines):
      try:
        obj = json.loads(json_lines[i])
        name = obj.get('skill_name', '')
        if name != skill.name:
          raise ValueError(f'Skill name mismatch: {name} != {skill.name}')
        rating = max(0, min(5, int(obj.get('rating', 0))))
        description = obj.get('description', '')
        description = description or skill.description
        results.append((rating, description))
      except (json.JSONDecodeError, ValueError, TypeError) as e:
        logging.warning(
            'Failed to parse LLM response for skill %s, line: %s, error: %s',
            skill.name,
            json_lines[i] if i < len(json_lines) else 'N/A',
            e,
        )
        results.append((0, skill.description))
    else:
      logging.warning(
          'No LLM response for skill %s (got %d lines, expected %d)',
          skill.name,
          len(json_lines),
          len(skills),
      )
      results.append((0, skill.description))

  return results


def load_skills(
    skills_dir: str,
    task: str,
    system_llm: llm_lib.LLMBase,
    max_skills: int = 10,
    all_skills: list[SkillInfo] | None = None,
) -> list[memory_lib.MemoryFile]:
  """Load and filter skills relevant to the given task.

  Args:
    skills_dir: Directory containing skill definitions.
    task: The task description, for judging skill relevance.
    system_llm: The LLM to use for relevance evaluation.
    max_skills: Maximum number of skills to load.
    all_skills: Pre-loaded skill files. If None, loads from skills_dir.

  Returns:
    A list of MemoryFile objects for the relevant skills.

  Raises:
    FileNotFoundError: If the skills directory does not exist.
  """
  console = rich.console.Console()
  skills_dir_path = epath.Path(skills_dir).expanduser()
  if not skills_dir_path.is_dir():
    console.print(
        f'[yellow]Skills directory not found: {skills_dir_path}[/yellow]'
    )
    return []

  # Step 1: Find and load all skill files
  if all_skills is None:
    console.print('\n[bold blue]Loading skills...[/bold blue]')
    all_skills = _load_skill_files(skills_dir_path)
  if not all_skills:
    console.print('[yellow]  No skills found.[/yellow]')
    return []
  console.print(f'  Found [bold]{len(all_skills)}[/bold] skills')

  # Step 2: Batch skills for relevance evaluation
  max_tokens_per_call = int(system_llm.max_tokens * 0.4)
  token_counter = lambda x: system_llm.count_tokens(
      [{'role': 'user', 'content': x}]
  )
  batches = _batch_skills(task, all_skills, max_tokens_per_call, token_counter)

  # Step 3: Evaluate relevance using system LLM
  rated_skills: list[RatedSkill] = []

  with rich.progress.Progress(
      rich.progress.SpinnerColumn(),
      rich.progress.TextColumn('[progress.description]{task.description}'),
      rich.progress.BarColumn(),
      rich.progress.TaskProgressColumn(),
      rich.progress.TimeElapsedColumn(),
      console=console,
  ) as progress:
    eval_task = progress.add_task(
        '  Evaluating skill relevance...', total=len(batches)
    )

    for batch_idx, batch in enumerate(batches):
      skill_names = ', '.join(s.name for s in batch)
      logging.info(
          'Evaluating batch %d/%d (%d skills: %s)',
          batch_idx + 1,
          len(batches),
          len(batch),
          skill_names,
      )

      prompt = _build_relevance_prompt(task, batch)
      messages = [{'role': 'user', 'content': prompt}]

      try:
        response = system_llm.completion(
            messages=messages,
            tools=[],
            system_prompt=_RELEVANCE_SYSTEM_PROMPT,
            num_retries=2,
        )
        ratings = _parse_llm_response(response.text, batch)
      except Exception as e:  # pylint: disable=broad-except
        logging.error('LLM call failed for batch %d: %s', batch_idx + 1, e)
        ratings = [(0, s.description) for s in batch]

      for skill, (rating, description) in zip(batch, ratings):
        rated_skills.append(
            RatedSkill(
                skill=skill,
                rating=rating,
                updated_description=description,
            )
        )

      progress.update(eval_task, advance=1)

  # Step 4: Sort by rating and select top-k
  rated_skills.sort(key=lambda rs: rs.rating, reverse=True)
  relevant_skills = [rs for rs in rated_skills if rs.rating > 0]

  if len(relevant_skills) > max_skills:
    relevant_skills = relevant_skills[:max_skills]

  table = rich.table.Table(title='Selected Skills')
  table.add_column('Rating', justify='center', style='bold')
  table.add_column('Name', style='cyan')
  table.add_column('Description', max_width=60)

  for rs in relevant_skills:
    stars = '⭐' * rs.rating
    table.add_row(stars, rs.skill.name, rs.updated_description[:80])

  console.print(table)
  console.print(
      f'\n  Selected [bold green]{len(relevant_skills)}[/bold green] '
      f'skills out of {len(all_skills)} total.\n'
  )

  memory_files: list[memory_lib.MemoryFile] = []
  for rs in relevant_skills:
    memory_files.append(
        memory_lib.MemoryFile(
            uri=f'kb://skill_{rs.skill.name}.md',
            content=rs.skill.content,
            summary=rs.updated_description,
            display=memory_lib.DisplayMode.SUMMARY,
            update_step=0,
        )
    )

  return memory_files


_CACHE_MAX_AGE_DAYS = 14


def _compute_skills_hash(
    task: str,
    all_skills: list[SkillInfo],
    max_skills: int,
) -> str:
  """Combined hash of the task, max_skills, and all loaded skill contents."""
  h = hashlib.sha256()
  h.update(task.encode())
  h.update(str(max_skills).encode())
  for skill in all_skills:
    h.update(skill.folder_name.encode())
    h.update(skill.content.encode())
  return h.hexdigest()


def _prune_old_caches(cache_dir: epath.Path) -> None:
  """Removes cache files older than _CACHE_MAX_AGE_DAYS days."""
  cutoff = datetime.datetime.now() - datetime.timedelta(
      days=_CACHE_MAX_AGE_DAYS
  )
  for cache_file in cache_dir.glob('skills_*.json'):
    try:
      # Filename format: skills_YYYYMMDD_<hash>.json
      date_str = cache_file.name.split('_')[1]
      file_date = datetime.datetime.strptime(date_str, '%Y%m%d')
      if file_date < cutoff:
        cache_file.unlink()
        logging.info('Pruned old skill cache: %s', cache_file)
    except (ValueError, IndexError):
      pass  # Skip files that don't match the expected format


def load_skills_cached(
    skills_dir: str,
    task: str,
    system_llm: llm_lib.LLMBase,
    max_skills: int = 10,
    cache_dir: epath.Path | None = None,
) -> list[memory_lib.MemoryFile]:
  """Load and filter skills with caching.

  Computes a hash of the skill files and task. If a matching cache exists,
  returns the cached result. Otherwise, runs the full skill loading pipeline
  and saves the result to cache.

  Args:
    skills_dir: Directory containing skill definitions.
    task: The task description, for judging skill relevance.
    system_llm: The LLM to use for relevance evaluation.
    max_skills: Maximum number of skills to load.
    cache_dir: Directory to store/read cache files. If None, caching is disabled
      and load_skills() is called directly.

  Returns:
    A list of MemoryFile objects for the relevant skills.

  Raises:
    FileNotFoundError: If the skills directory is not found.
    ValueError: If the cache file is corrupted.
  """
  console = rich.console.Console()

  if cache_dir is None:
    return load_skills(
        skills_dir=skills_dir,
        task=task,
        system_llm=system_llm,
        max_skills=max_skills,
    )

  # Load all skill files first (used for both hashing and evaluation)
  console.print('\n[bold blue]Loading skills...[/bold blue]')
  skills_dir_path = epath.Path(skills_dir).expanduser()
  if not skills_dir_path.is_dir():
    console.print(
        f'[yellow]Skills directory not found: {skills_dir_path}[/yellow]'
    )
    return []

  all_skills = _load_skill_files(skills_dir_path)
  if not all_skills:
    console.print('[yellow]  No skills found.[/yellow]')
    return []

  cache_dir.mkdir(parents=True, exist_ok=True)
  _prune_old_caches(cache_dir)

  skills_hash = _compute_skills_hash(task, all_skills, max_skills)
  date_str = datetime.datetime.now().strftime('%Y%m%d')

  # Check for any existing cache file with matching hash
  cached_file = None
  for f in cache_dir.glob(f'skills_*_{skills_hash}.json'):
    cached_file = f
    break

  if cached_file is not None:
    console.print(
        '\n[bold green]Skill cache hit![/bold green] Loading from'
        f' {cached_file}'
    )
    try:
      data = json.loads(cached_file.read_text())
      memory_files = [
          memory_lib.MemoryFile(
              uri=entry['uri'],
              content=entry['content'],
              summary=entry['summary'],
              display=memory_lib.DisplayMode(entry['display']),
              update_step=0,
          )
          for entry in data['selected_skills']
      ]
      console.print(
          f'  Loaded [bold]{len(memory_files)}[/bold] skills from cache.\n'
      )
      return memory_files
    except Exception as e:  # pylint: disable=broad-except
      logging.warning(
          'Failed to load skill cache %s: %s. Recomputing.', cached_file, e
      )

  # Cache miss: run the full pipeline with pre-loaded skills
  memory_files = load_skills(
      skills_dir=skills_dir,
      task=task,
      system_llm=system_llm,
      max_skills=max_skills,
      all_skills=all_skills,
  )

  # Save to cache
  cache_filename = f'skills_{date_str}_{skills_hash}.json'
  cache_path = cache_dir / cache_filename
  try:
    data = {
        'hash': skills_hash,
        'date': date_str,
        'max_skills': max_skills,
        'selected_skills': [
            {
                'uri': mf.uri,
                'content': mf.content,
                'summary': mf.summary,
                'display': mf.display.value,
            }
            for mf in memory_files
        ],
    }
    cache_path.write_text(json.dumps(data, indent=2))
    console.print(f'  Skill cache saved to {cache_path}\n')
  except Exception as e:  # pylint: disable=broad-except
    logging.warning('Failed to save skill cache: %s', e)

  return memory_files
