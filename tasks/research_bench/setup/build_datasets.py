# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stages the research-bench evaluation/training datasets.

Every builder writes the exact on-disk shape the (ported) data source reads,
and every builder ends with a check on the example count and field schema, so
a silent upstream reshuffle fails the build instead of the experiment.

Expected counts (asserted):

| file                                    | rows  | read by                    |
|-----------------------------------------|-------|----------------------------|
| gsm8k/gsm8k.json                        | 7473 + 1319 | simply:gsm8k_{train,test} |
| math500/test.json                       | 500 (262 at L4-5) | simply:math500_test_l45 |
| deepscaler/deepscaler.json              | 40315 | simply:dsr40k_train        |
| aime/aime_v3.json                       | 1035 (30 in 2025) | simply:aime25  |
| tooluse_rlvr/bfcl/<category>.json       | see BFCL_COUNTS | simply:bfcl_*        |
| tooluse_rlvr/toolace/toolace_bfcl.json  | ~9.3k | simply:toolace_train       |
| livecodebench/livecodebench_v5.json     | 167   | simply_json:livecodebench_v5 |
"""

from __future__ import annotations

import ast
import base64
import json
import os
import pickle
import re
import zlib
from typing import Any, Sequence

from tasks.research_bench.setup import asset_lib

D = asset_lib.DATASETS_DIR

# ---------------------------------------------------------------------------
# Sources (repo ids pinned by name; revisions recorded in the manifest).
# ---------------------------------------------------------------------------
SIMPLY_DATASETS_REPO = 'unkindledmonkey/simply-datasets'
MATH500_REPO = 'HuggingFaceH4/MATH-500'
DEEPSCALER_REPO = 'agentica-org/DeepScaleR-Preview-Dataset'
BFCL_REPO = 'gorilla-llm/Berkeley-Function-Calling-Leaderboard'
TOOLACE_REPO = 'Team-ACE/ToolACE'
LCB_REPO = 'livecodebench/code_generation_lite'
AIME25_REPO = 'opencompass/AIME2025'

# The BFCL AST categories the tool-use task uses, with their v3 sizes. The
# non-live five are the RL train split; the four call-required live ones are
# the FIXED held-out eval (1351 problems). `*irrelevance` has no
# possible_answer file (abstention: ground_truth is None).
BFCL_COUNTS = {
    'simple': 400,
    'multiple': 200,
    'parallel': 200,
    'parallel_multiple': 200,
    'irrelevance': 240,
    'live_simple': 258,
    'live_multiple': 1053,
    'live_parallel': 16,
    'live_parallel_multiple': 24,
    'live_irrelevance': 882,
}
BFCL_NO_ANSWER_CATEGORIES = ('irrelevance', 'live_irrelevance')

# LiveCodeBench "v5" as used by the sampling_lcb task: the v4 -> v5 increment,
# which `code_generation_lite` ships as its own file (HF loader config `v5`)
# and which is exactly the 167 problems (2024-09-22 .. 2025-01-04) the task
# scores. Verified byte-identical to the internal copy the task reads.
LCB_V5_FILE = 'test5.jsonl'
LCB_V5_SHA256 = (
    '7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214'
)
LCB_V5_COUNT = 167
LCB_V5_PUBLIC_TESTS = 441
LCB_V5_PRIVATE_TESTS = 6099


def _load_jsonl(path: str) -> list[dict[str, Any]]:
  return asset_lib.read_jsonl(path)


def _require(condition: bool, msg: str) -> None:
  if not condition:
    raise ValueError(f'dataset check failed: {msg}')


# ---------------------------------------------------------------------------
# gsm8k / MATH500 / DeepScaleR / AIME: plain question-answer JSON.
# ---------------------------------------------------------------------------
def build_gsm8k(force: bool = False) -> str:
  """Stages `gsm8k/gsm8k.json` (dict of splits) from the simply-datasets repo."""
  dst = os.path.join(D, 'gsm8k/gsm8k.json')
  if force or not os.path.exists(dst):
    src = asset_lib.hf_snapshot(
        SIMPLY_DATASETS_REPO, repo_type='dataset', allow_patterns=['gsm8k/*']
    )
    for name in ('gsm8k.json', 'LICENSE'):
      if os.path.exists(os.path.join(src, 'gsm8k', name)):
        asset_lib.stage_file(
            os.path.join(src, 'gsm8k', name),
            os.path.join(D, 'gsm8k', name),
            overwrite=force,
        )
  with open(dst, encoding='utf-8') as f:
    data = json.load(f)
  _require(len(data['train']) == 7473, f'gsm8k train {len(data["train"])}')
  _require(len(data['test']) == 1319, f'gsm8k test {len(data["test"])}')
  _require(
      {'question', 'answer'} <= set(data['test'][0]),
      f'gsm8k fields {sorted(data["test"][0])}',
  )
  return dst


def build_math500(force: bool = False) -> str:
  """Stages `math500/test.json` (the 500 MATH test problems, all levels).

  `force` is accepted for a uniform builder signature; this one always
  rewrites.

  `simply:math500_test_l45` filters to levels 4-5 (262 problems) at read time,
  so the staged file stays the unmodified upstream test split.
  """
  dst = os.path.join(D, 'math500/test.json')
  rows = _load_jsonl(
      asset_lib.hf_file(MATH500_REPO, 'test.jsonl', repo_type='dataset')
  )
  _require(len(rows) == 500, f'MATH-500 rows {len(rows)}')
  fields = {'problem', 'answer', 'solution', 'subject', 'level', 'unique_id'}
  _require(fields <= set(rows[0]), f'MATH-500 fields {sorted(rows[0])}')
  n45 = sum(1 for r in rows if r['level'] in (4, 5))
  _require(n45 == 262, f'MATH-500 levels 4-5 = {n45}, expected 262')
  asset_lib.write_json(rows, dst)
  return dst


def build_deepscaler(force: bool = False) -> str:
  """Stages `deepscaler/deepscaler.json` (the RL train split for the math task)."""
  dst = os.path.join(D, 'deepscaler/deepscaler.json')
  src = asset_lib.hf_file(
      DEEPSCALER_REPO, 'deepscaler.json', repo_type='dataset'
  )
  with open(src, encoding='utf-8') as f:
    rows = json.load(f)
  _require(
      {'problem', 'answer', 'solution'} <= set(rows[0]),
      f'deepscaler fields {sorted(rows[0])}',
  )
  asset_lib.stage_file(src, dst, overwrite=True)
  return dst


def build_aime(force: bool = False) -> str:
  """Stages `aime/aime_v3.json`, the year-tagged AIME archive incl. 2025.

  `simply:aime25` selects `year == 2025` (30 problems: AIME 2025 I + II).
  """
  dst = os.path.join(D, 'aime/aime_v3.json')
  src_dir = asset_lib.hf_snapshot(
      SIMPLY_DATASETS_REPO, repo_type='dataset', allow_patterns=['aime/*']
  )
  archive = os.path.join(src_dir, 'aime', 'aime_v3.json')
  if not os.path.exists(archive):
    archive = os.path.join(src_dir, 'aime', 'aime_v2.json')
  with open(archive, encoding='utf-8') as f:
    rows = json.load(f)
  extra = [] if any(int(r['year']) == 2025 for r in rows) else _aime2025_rows(
      len(rows)
  )
  n25 = sum(1 for r in rows + extra if int(r['year']) == 2025)
  _require(n25 == 30, f'AIME 2025 problems = {n25}, expected 30')
  _require(
      {'problem', 'answer', 'solution', 'year'} <= set(rows[0]),
      f'aime fields {sorted(rows[0])}',
  )
  if extra:
    asset_lib.write_json(rows + extra, dst)
  else:
    asset_lib.stage_file(archive, dst, overwrite=True)
  return dst


def _aime2025_rows(offset: int) -> list[dict[str, Any]]:
  """AIME 2025 I+II from `opencompass/AIME2025` in the archive's schema."""
  rows = []
  for part, name in enumerate(('aime2025-I.jsonl', 'aime2025-II.jsonl')):
    for i, r in enumerate(
        _load_jsonl(asset_lib.hf_file(AIME25_REPO, name, repo_type='dataset'))
    ):
      rows.append({
          'problem': r['question'],
          'answer': str(r['answer']),
          'solution': r.get('solution', ''),
          'year': 2025,
          'aime_number': part + 1,
          'problem_number': i + 1,
          'id': offset + len(rows),
      })
  return rows


# ---------------------------------------------------------------------------
# BFCL (AST categories) -> tooluse_rlvr/bfcl/<category>.json
# ---------------------------------------------------------------------------
def build_bfcl(
    categories: Sequence[str] = tuple(BFCL_COUNTS), force: bool = False
) -> list[str]:
  """Stages one JSON array per BFCL category, ground truth merged in.

  The staged schema is the upstream one plus `ground_truth`:
  `{id, question, function, ground_truth}`, which is exactly what the ported
  `BFCLSource` reads (it json-dumps the nested fields itself).

  Args:
    categories: BFCL AST categories to stage.
    force: unused; this builder always rewrites.

  Returns:
    The staged file paths.
  """
  patterns = [f'BFCL_v3_{c}.json' for c in categories]
  patterns += [f'possible_answer/BFCL_v3_{c}.json' for c in categories]
  src = asset_lib.hf_snapshot(
      BFCL_REPO, repo_type='dataset', allow_patterns=patterns
  )
  out = []
  for cat in categories:
    rows = _load_jsonl(os.path.join(src, f'BFCL_v3_{cat}.json'))
    _require(
        len(rows) == BFCL_COUNTS[cat],
        f'bfcl {cat}: {len(rows)} rows, expected {BFCL_COUNTS[cat]}',
    )
    answer_path = os.path.join(src, 'possible_answer', f'BFCL_v3_{cat}.json')
    if os.path.exists(answer_path):
      # Paired by line order, as the upstream BFCL evaluator does: a handful of
      # v3 ids disagree between the two files (e.g. the question
      # `live_multiple_1052-79-0` is answered as `live_multiple_1052-279-0`).
      answers = _load_jsonl(answer_path)
      _require(
          len(answers) == len(rows),
          f'bfcl {cat}: {len(answers)} answers for {len(rows)} rows',
      )
      for row, answer in zip(rows, answers):
        if row['id'] != answer['id']:
          asset_lib.log(
              f'bfcl {cat}: id mismatch at line order'
              f' {row["id"]} vs {answer["id"]} (pairing by order)'
          )
        row['ground_truth'] = answer['ground_truth']
    else:
      _require(
          cat in BFCL_NO_ANSWER_CATEGORIES,
          f'bfcl {cat}: missing possible_answer file',
      )
      for row in rows:
        row['ground_truth'] = None
    for row in rows:
      _require(
          {'id', 'question', 'function'} <= set(row),
          f'bfcl {cat} fields {sorted(row)}',
      )
    out.append(
        asset_lib.write_json(rows, os.path.join(D, f'tooluse_rlvr/bfcl/{cat}.json'))
    )
  return out


# ---------------------------------------------------------------------------
# ToolACE -> the same BFCL example schema.
# ---------------------------------------------------------------------------
_TOOLACE_FUNCTIONS_RE = re.compile(r'\[\s*{.*}\s*\]', re.DOTALL)


def _toolace_functions(system: str) -> list[dict[str, Any]] | None:
  """Extracts the JSON tool schemas embedded in a ToolACE system prompt."""
  match = _TOOLACE_FUNCTIONS_RE.search(system)
  if not match:
    return None
  try:
    functions = json.loads(match.group(0))
  except json.JSONDecodeError:
    return None
  return functions if isinstance(functions, list) else None


def _split_calls(text: str) -> list[tuple[str, str]] | None:
  """Splits `[Name(args), Other Name(args)]` into `(name, arg_string)` pairs.

  Hand-written rather than `ast.parse`d because ToolACE tool names contain
  spaces (`Market Trends API(...)`), which is not valid Python.

  Args:
    text: the assistant turn.

  Returns:
    The calls, or None if `text` is not a well-formed call list.
  """
  text = text.strip()
  if not (text.startswith('[') and text.endswith(']')):
    return None
  body = text[1:-1].strip()
  calls, i, n = [], 0, len(body)
  while i < n:
    open_paren = body.find('(', i)
    if open_paren < 0:
      return None
    name = body[i:open_paren].strip()
    if not name:
      return None
    depth, k, quote = 1, open_paren + 1, ''
    while k < n and depth:
      char = body[k]
      if quote:
        if char == '\\':
          k += 1
        elif char == quote:
          quote = ''
      elif char in '"\'':
        quote = char
      elif char == '(':
        depth += 1
      elif char == ')':
        depth -= 1
      k += 1
    if depth:
      return None
    calls.append((name, body[open_paren + 1 : k - 1]))
    while k < n and body[k] in ', \n\t':
      k += 1
    i = k
  return calls or None


def _parse_kwargs(arg_string: str) -> dict[str, Any] | None:
  """Parses a call's argument list; None unless every value is a literal."""
  try:
    call = ast.parse(f'_f({arg_string})', mode='eval').body
  except SyntaxError:
    return None
  if not isinstance(call, ast.Call) or call.args:
    return None
  kwargs = {}
  for keyword in call.keywords:
    if keyword.arg is None:
      return None
    try:
      kwargs[keyword.arg] = ast.literal_eval(keyword.value)
    except (ValueError, SyntaxError):
      return None
  return kwargs


def build_toolace(force: bool = False) -> str:
  """Converts ToolACE to BFCL `{question, function, ground_truth}` rows.

  Keeps the single-user-turn rows whose assistant reply is a pure
  literal-argument call list over functions declared in the system prompt; a
  reference call becomes the BFCL allowed-value set `{arg: [value]}`.
  """
  dst = os.path.join(D, 'tooluse_rlvr/toolace/toolace_bfcl.json')
  with open(
      asset_lib.hf_file(TOOLACE_REPO, 'data.json', repo_type='dataset'),
      encoding='utf-8',
  ) as f:
    raw = json.load(f)
  out = []
  for i, row in enumerate(raw):
    turns = row.get('conversations') or []
    if len(turns) < 2 or turns[0]['from'] != 'user':
      continue
    if turns[1]['from'] != 'assistant':
      continue
    functions = _toolace_functions(row.get('system', ''))
    if not functions:
      continue
    calls = _split_calls(turns[1]['value'])
    if not calls:
      continue
    declared = {fn.get('name') for fn in functions}
    ground_truth = []
    for name, arg_string in calls:
      kwargs = _parse_kwargs(arg_string) if name in declared else None
      if kwargs is None:
        ground_truth = []
        break
      ground_truth.append({name: {k: [v] for k, v in kwargs.items()}})
    if not ground_truth:
      continue
    out.append({
        'id': f'toolace_{i}',
        'question': [[{'role': 'user', 'content': turns[0]['value']}]],
        'function': functions,
        'ground_truth': ground_truth,
    })
  _require(len(out) > 8000, f'toolace kept only {len(out)} rows')
  asset_lib.write_json(out, dst)
  return dst


# ---------------------------------------------------------------------------
# LiveCodeBench v5 -> livecodebench/livecodebench_v5.json
# ---------------------------------------------------------------------------
def _decode_tests(blob: str) -> list[dict[str, Any]]:
  """Decodes LCB test cases, which may be base64(zlib(pickle(json)))."""
  try:
    return json.loads(blob)
  except json.JSONDecodeError:
    return json.loads(
        pickle.loads(zlib.decompress(base64.b64decode(blob.encode('utf-8'))))
    )


def build_livecodebench(force: bool = False) -> str:
  """Stages the 167 LiveCodeBench v5 problems with their tests decoded.

  Written as JSONL in upstream file order: the decoded private tests total
  ~1.15 GB, so the loader indexes line offsets and parses one problem at a
  time instead of holding the whole list in memory. Each row keeps the
  upstream fields the grader needs plus decoded `public_test_cases` /
  `private_test_cases` (JSON strings) and `func_name`, so reading needs
  neither `pickle` nor the `datasets` package.
  """
  dst = os.path.join(D, 'livecodebench/livecodebench_v5.jsonl')
  # The only builder worth an early-out: decoding the private tests takes ~20 s
  # and writes 1.1 GiB. `check()` still re-reads it.
  if not force and os.path.exists(dst) and sum(
      1 for _ in open(dst, encoding='utf-8')
  ) == LCB_V5_COUNT:
    return dst
  path = asset_lib.hf_file(LCB_REPO, LCB_V5_FILE, repo_type='dataset')
  got = asset_lib.sha256(path)
  _require(
      got == LCB_V5_SHA256, f'{LCB_V5_FILE}: sha256 {got} != {LCB_V5_SHA256}'
  )
  rows = _load_jsonl(path)
  _require(
      len(rows) == LCB_V5_COUNT,
      f'livecodebench v5: {len(rows)} problems, expected {LCB_V5_COUNT}',
  )
  asset_lib.ensure_dir(os.path.dirname(dst))
  public = private = 0
  with open(dst + '.part', 'w', encoding='utf-8') as out:
    for row in rows:
      metadata = json.loads(row['metadata']) if row.get('metadata') else {}
      public_tests = _decode_tests(row['public_test_cases'])
      private_tests = _decode_tests(row['private_test_cases'])
      public += len(public_tests)
      private += len(private_tests)
      out.write(json.dumps({
          'question_id': row['question_id'],
          'question_title': row.get('question_title', ''),
          'question_content': row['question_content'],
          'platform': row.get('platform', ''),
          'contest_id': row.get('contest_id', ''),
          'contest_date': row['contest_date'],
          'difficulty': row.get('difficulty', ''),
          'starter_code': row.get('starter_code', ''),
          'func_name': metadata.get('func_name', ''),
          'public_test_cases': json.dumps(public_tests),
          'private_test_cases': json.dumps(private_tests),
      }) + '\n')
  _require(
      (public, private) == (LCB_V5_PUBLIC_TESTS, LCB_V5_PRIVATE_TESTS),
      f'livecodebench v5 tests: {public} public / {private} private',
  )
  os.replace(dst + '.part', dst)
  return dst


# ---------------------------------------------------------------------------
# Loader checks: read every staged file the way the task code will.
# ---------------------------------------------------------------------------
def check() -> dict[str, Any]:
  """Re-reads every staged dataset and returns a summary (raises on mismatch)."""
  summary: dict[str, Any] = {}

  with open(os.path.join(D, 'gsm8k/gsm8k.json'), encoding='utf-8') as f:
    gsm = json.load(f)
  summary['gsm8k'] = {'train': len(gsm['train']), 'test': len(gsm['test'])}
  _require(summary['gsm8k'] == {'train': 7473, 'test': 1319}, 'gsm8k counts')

  with open(os.path.join(D, 'math500/test.json'), encoding='utf-8') as f:
    math500 = json.load(f)
  l45 = [r for r in math500 if r['level'] in (4, 5)]
  summary['math500'] = {'all': len(math500), 'level_4_5': len(l45)}
  _require(len(l45) == 262, 'math500 L4-5 != 262')

  with open(os.path.join(D, 'deepscaler/deepscaler.json'), encoding='utf-8') as f:
    summary['deepscaler'] = len(json.load(f))

  with open(os.path.join(D, 'aime/aime_v3.json'), encoding='utf-8') as f:
    aime = json.load(f)
  summary['aime25'] = sum(1 for r in aime if int(r['year']) == 2025)
  _require(summary['aime25'] == 30, 'aime25 != 30')

  bfcl = {}
  for cat in BFCL_COUNTS:
    path = os.path.join(D, f'tooluse_rlvr/bfcl/{cat}.json')
    if not os.path.exists(path):
      continue
    with open(path, encoding='utf-8') as f:
      rows = json.load(f)
    bfcl[cat] = len(rows)
    _require(len(rows) == BFCL_COUNTS[cat], f'bfcl {cat} count')
    if cat not in BFCL_NO_ANSWER_CATEGORIES:
      _require(
          all(r.get('ground_truth') for r in rows), f'bfcl {cat} ground_truth'
      )
  summary['bfcl'] = bfcl
  summary['bfcl_live_call_required'] = sum(
      bfcl.get(c, 0)
      for c in ('live_simple', 'live_multiple', 'live_parallel',
                'live_parallel_multiple')
  )

  toolace_path = os.path.join(D, 'tooluse_rlvr/toolace/toolace_bfcl.json')
  if os.path.exists(toolace_path):
    with open(toolace_path, encoding='utf-8') as f:
      summary['toolace'] = len(json.load(f))

  lcb_path = os.path.join(D, 'livecodebench/livecodebench_v5.jsonl')
  if os.path.exists(lcb_path):
    public = private = count = 0
    first = None
    with open(lcb_path, encoding='utf-8') as f:
      for line in f:
        row = json.loads(line)
        first = first or row
        count += 1
        public += len(json.loads(row['public_test_cases']))
        private += len(json.loads(row['private_test_cases']))
    summary['livecodebench_v5'] = {
        'problems': count,
        'public_tests': public,
        'private_tests': private,
        'first_question_id': first['question_id'] if first else '',
    }
    _require(count == LCB_V5_COUNT, 'lcb v5 count')
    _require(
        (public, private) == (LCB_V5_PUBLIC_TESTS, LCB_V5_PRIVATE_TESTS),
        'lcb v5 test counts',
    )
  return summary
