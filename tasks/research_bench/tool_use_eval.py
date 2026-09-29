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

"""BFCL function-calling RLVR evaluation (AST-based verifiable reward).

Registers `BFCLFunctionCallEvaluation` into core simply's EvaluationRegistry so
the `tool_use_bfcl_rl` task can use it for both the RL training reward and the
held-out eval. The reward parses the model's emitted call list `[f(a=1), ...]`
and structurally checks name + args against BFCL ground-truth allowed-value sets
-- no code execution / sandbox. This is the FIXED evaluation for the task and
must not be modified by the agent.
"""

import ast
import dataclasses
import json
import re
from typing import Any, Mapping, Sequence

from simply.utils import evaluation_lib as _eval_lib
from simply.utils import lm_format as lm_format_lib  # pylint: disable=unused-import

EvaluationRegistry = _eval_lib.EvaluationRegistry
Evaluation = _eval_lib.Evaluation

# ------------------------------------------------------------------------------
# BFCL function-calling RLVR (AST-based verifiable reward).
# ------------------------------------------------------------------------------
_BFCL_UNPARSEABLE = object()


def _bfcl_name_of(node) -> str | None:
  """Resolves a (possibly dotted) call name, e.g. `math.factorial`."""
  if isinstance(node, ast.Name):
    return node.id
  if isinstance(node, ast.Attribute):
    base = _bfcl_name_of(node.value)
    return f'{base}.{node.attr}' if base else None
  return None


def _first_bracketed_list(text: str) -> str | None:
  r"""Returns the first top-level `[...]` span by bracket-depth matching.

  A greedy `\[.*\]` would run from the first `[` to the LAST `]`, so a
  correct-then-verbose completion (e.g. the call list followed by prose that
  contains another `]`) would fail to parse -> false negative. We instead take
  the first `[` and its depth-matched `]` (ignoring brackets inside string
  literals), which is the emitted call list.

  Args:
    text: the raw model completion text to scan.

  Returns:
    The first top-level `[...]` span, or None if no balanced list is found.
  """
  start = text.find('[')
  if start < 0:
    return None
  depth = 0
  in_str = None  # current string-quote char, or None
  esc = False
  for i in range(start, len(text)):
    c = text[i]
    if in_str is not None:
      if esc:
        esc = False
      elif c == '\\':
        esc = True
      elif c == in_str:
        in_str = None
      continue
    if c in ('"', "'"):
      in_str = c
    elif c == '[':
      depth += 1
    elif c == ']':
      depth -= 1
      if depth == 0:
        return text[start : i + 1]
  return None  # unbalanced


def bfcl_parse_calls(response: str):
  """Parses `[f(a=1), g(b=2)]` -> [(name, {arg: val}), ...]; None if unparsable."""
  segment = _first_bracketed_list(response.strip())
  if segment is None:
    return None
  try:
    tree = ast.parse(segment, mode='eval')
  # Model output is arbitrary text; parsing it must NEVER crash the run. Catch
  # broadly (e.g. ast.parse can raise SyntaxError/ValueError/MemoryError/...).
  except Exception:  # pylint: disable=broad-except
    return None
  if not isinstance(tree.body, ast.List):
    return None
  calls = []
  for elt in tree.body.elts:
    if not isinstance(elt, ast.Call):
      return None
    name = _bfcl_name_of(elt.func)
    if name is None:
      return None
    kwargs = {}
    for kw in elt.keywords:
      if kw.arg is None:  # **kwargs unpacking, not allowed
        return None
      try:
        kwargs[kw.arg] = ast.literal_eval(kw.value)
      # literal_eval can raise beyond ValueError/SyntaxError, e.g. TypeError on
      # a set literal containing an unhashable value like `{ {..}: .. }` /
      # `{ {..} }`. Treat any failure as an unparseable arg (never crash).
      except Exception:  # pylint: disable=broad-except
        kwargs[kw.arg] = _BFCL_UNPARSEABLE
    calls.append((name, kwargs))
  return calls


# ---------------------------------------------------------------------------
# Faithful port of the upstream Gorilla BFCL AST checker (Python path), pinned
# to the same commit the data comes from. Ported so our verifiable reward
# matches the official leaderboard semantics: type-aware matching (no int/str
# leniency), dict/list/list-of-dict checkers, and standardize_string
# (punctuation/space-insensitive) value comparison.
# ---------------------------------------------------------------------------
_PY_TYPE = {
    'string': str,
    'integer': int,
    'float': float,
    'boolean': bool,
    'array': list,
    'tuple': list,
    'dict': dict,
    'any': str,
}
# Non-canonical spellings seen in third-party function-calling corpora (e.g.
# ToolACE uses 'int'/'str'/'bool'; JSON-Schema proper uses 'number'/'object').
# Mapped to the canonical BFCL type names so the type check stays STRICT instead
# of silently falling back to `str`. The BFCL data itself (all 8 AST categories,
# train + live eval) uses only canonical names, so this is a no-op there --
# verified against the staged JSON; it only affects extra training corpora.
_PY_TYPE_ALIASES = {
    'str': 'string',
    'int': 'integer',
    'number': 'float',
    'double': 'float',
    'bool': 'boolean',
    'list': 'array',
    'object': 'dict',
}
_NESTED_TYPES = ('array', 'tuple')
_STD_RE = re.compile(r'[ \,\.\/\-\_\*\^]')


def _standardize_string(s: str) -> str:
  return _STD_RE.sub('', s).lower().replace("'", '"')


def _string_match(model_output, allowed) -> bool:
  if not isinstance(model_output, str):
    return False
  smo = _standardize_string(model_output)
  return smo in [_standardize_string(a) for a in allowed if isinstance(a, str)]


def _list_match(model_output, allowed) -> bool:
  """model_output is a list; allowed is a list of acceptable list values."""
  std = [
      _standardize_string(x) if isinstance(x, str) else x for x in model_output
  ]
  for ans in allowed:
    if not isinstance(ans, (list, tuple)):
      continue
    std_ans = [_standardize_string(x) if isinstance(x, str) else x for x in ans]
    if std == std_ans:
      return True
  return False


def _acceptable_values(v):
  """Returns the acceptable-value list for one argument.

  BFCL ground truth maps each argument to a LIST of acceptable values, but a
  dict-valued argument nests a plain call dict whose leaves are bare scalars
  ({'mode': 'fast'}, not {'mode': ['fast']}). Iterating such a leaf raises
  TypeError on numbers and silently iterates CHARACTERS on strings, so a nested
  dict argument could never match. Treat a non-list leaf as one acceptable
  value.

  Args:
    v: an entry of a ground-truth argument dict.

  Returns:
    v itself when it is already a sequence of acceptable values, else [v].
  """
  return v if isinstance(v, (list, tuple)) else [v]


def _dict_match(model_output, allowed) -> bool:
  """model_output is a dict; allowed is a list of acceptable {k:[vals]} dicts."""
  if not isinstance(model_output, dict):
    return False
  for possible in allowed:
    if not isinstance(possible, dict):
      continue
    ok = True
    for key, value in model_output.items():
      if key not in possible:
        ok = False
        break
      sval = _standardize_string(value) if isinstance(value, str) else value
      allowed_vals = [
          _standardize_string(x) if isinstance(x, str) else x
          for x in _acceptable_values(possible[key])
      ]
      if sval not in allowed_vals:
        ok = False
        break
    if ok:
      for key, vals in possible.items():
        if key not in model_output and '' not in _acceptable_values(vals):
          ok = False
          break
    if ok:
      return True
  return False


def _list_dict_match(model_output, allowed) -> bool:
  """model_output is a list of dicts; order must match a possible-answer list."""
  if not isinstance(model_output, list):
    return False
  for ans in allowed:
    if not isinstance(ans, list) or len(ans) != len(model_output):
      continue
    if all(_dict_match(model_output[i], [ans[i]]) for i in range(len(ans))):
      return True
  return False


def _normalize_type(declared) -> str:
  """Canonicalizes a JSON-Schema `type` declaration to a `_PY_TYPE` key.

  Robust to the two shapes that are NOT a plain canonical string and would
  otherwise break the checker:
    * a UNION type (`{"type": ["string", "integer"]}`, legal JSON Schema) --
      unhashable, so `_PY_TYPE.get(...)` used to raise
      `TypeError: unhashable type: 'list'` and abort the whole run. Occurs in
      the shipped `simply:toolace_train` source (57/8523 rows). We take the
      first member that names a known type (upstream BFCL has no union support;
      this keeps the strictest interpretation we can express).
    * an ALIAS (`int`, `str`, `bool`, `number`, `object`, ...), which used to
      fall through to the `str` default and silently DISABLE type strictness.

  Anything unrecognized becomes 'any' (the upstream typeless path), never a
  crash.

  Args:
    declared: the raw `type` value from the tool schema (any JSON value).

  Returns:
    A canonical `_PY_TYPE` key, or 'any' when the declaration is unusable.
  """
  if isinstance(declared, str):
    t = declared.strip().lower()
    t = _PY_TYPE_ALIASES.get(t, t)
    return t if t in _PY_TYPE else 'any'
  if isinstance(declared, (list, tuple)):
    for item in declared:
      t = _normalize_type(item)
      if t != 'any':
        return t
  return 'any'


def _possible_answer_type(allowed):
  """Type of the first non-'' allowed value (upstream get_possible_answer_type)."""
  for a in allowed:
    if a != '':  # pylint: disable=g-explicit-bool-comparison
      return type(a)
  return None


def _value_matches(value, allowed, expected_type: str) -> bool:
  """Type-aware value match against the allowed set (upstream semantics)."""
  # `expected_type` is already canonicalized by `_normalize_type` at the call
  # site; re-normalize defensively so a direct caller cannot crash the run on an
  # unhashable (union) declaration.
  expected_type = _normalize_type(expected_type)
  expected = _PY_TYPE.get(expected_type, str)
  # tuple normalization (json round-trips tuples to lists).
  if expected_type == 'tuple' and isinstance(value, tuple):
    value = list(value)
  # upstream-sanctioned int->float widening.
  if (
      expected_type == 'float'
      and isinstance(value, int)
      and not isinstance(value, bool)
  ):
    value = float(value)
  # STRICT declared-type check (bool is a subclass of int, so guard explicitly).
  if expected is bool:
    type_ok = isinstance(value, bool)
  elif expected in (int, float):
    type_ok = isinstance(value, expected) and not isinstance(value, bool)
  else:
    type_ok = type(value) is expected  # pylint: disable=unidiomatic-typecheck
  # upstream 'is_variable' path: if the value's type doesn't match the declared
  # type but DOES match the type of the allowed set (e.g. None for an optional
  # arg whose allowed set is ['', None]), accept by raw membership. This is very
  # common in BFCL optional args -- omitting it rejects correct calls.
  if not type_ok:
    pat = _possible_answer_type(allowed)
    if pat is not None and type(value) is pat:  # pylint: disable=unidiomatic-typecheck
      return value in allowed
    return False
  # dispatch to the structured checkers.
  if expected is dict:
    return _dict_match(value, allowed)
  if expected is list:
    # list-of-dict if the allowed answers are lists of dicts.
    if any(
        isinstance(a, list) and a and isinstance(a[0], dict) for a in allowed
    ):
      return _list_dict_match(value, allowed)
    return _list_match(value, allowed)
  if expected is str:
    return _string_match(value, allowed)
  return value in allowed  # int / float / bool exact membership.


def _single_call_match(pred_name, pred_args, gt_call, schema_by_name) -> bool:
  """Match one predicted call vs one ground-truth call (upstream semantics).

  schema_by_name maps function name -> its schema dict ({name, parameters:{
  properties, required}}). Falls back to lenient/typeless matching only if the
  schema for the function is unavailable.

  Args:
    pred_name: predicted function name.
    pred_args: predicted argument dict.
    gt_call: the ground-truth call ({name: {arg: [allowed vals]}}).
    schema_by_name: map function name -> its schema dict.

  Returns:
    True iff the predicted call matches the ground truth under the AST checker.
  """
  ((gt_name, gt_args),) = gt_call.items()
  if pred_name != gt_name:
    return False
  schema = schema_by_name.get(gt_name)
  props = {}
  required = []
  if schema is not None:
    params = schema.get('parameters', {}) or {}
    props = params.get('properties', {}) or {}
    if isinstance(props, str):
      try:
        props = json.loads(props)
      except (ValueError, TypeError):
        props = {}
    required = params.get('required', []) or []
  # required params present.
  for param in required:
    if param not in pred_args:
      return False
  # each provided param: no hallucination, type+value match vs allowed set.
  for param, value in pred_args.items():
    if param not in gt_args:
      return False
    if value is _BFCL_UNPARSEABLE:
      return False
    allowed = gt_args[param]
    expected_type = _normalize_type(
        props.get(param, {}).get('type', 'any')
        if isinstance(props.get(param), dict)
        else 'any'
    )
    if not _value_matches(value, allowed, expected_type):
      return False
  # optional params: any gt param not provided must allow omission ('' in set).
  for param, allowed in gt_args.items():
    if param not in pred_args and '' not in allowed:
      return False
  return True


# Two-shot exemplar (one single-call, one abstention) shown before the user turn
# so a base/PT model learns the `Calls: [..]` output format and, crucially,
# to STOP right after the closing bracket instead of rambling into truncation.
BFCL_FEW_SHOT = (
    'Tools:\n'
    '- get_weather(city: string (required), unit: string (optional)): '
    'Get the current weather for a city.\n\n'
    'User: What is the weather in Paris in celsius?\n'
    'Calls: [get_weather(city="Paris", unit="celsius")]\n\n'
    'Tools:\n'
    '- add_numbers(a: integer (required), b: integer (required)): Add two '
    'integers.\n\n'
    'User: Book me a flight to Tokyo.\n'
    'Calls: []\n\n'
)


@EvaluationRegistry.register
@dataclasses.dataclass(frozen=True)
class BFCLFunctionCallEvaluation(Evaluation):
  """AST-based verifiable reward for BFCL function calling.

  Prompt: system instructions + tool schemas + user turn(s); the model must emit
  a Python-list of calls `[f(a=1), g(b=2)]` (or `[]` to abstain). The reward
  parses the emitted call(s) and structurally checks name + args against the
  BFCL `ground_truth` allowed-value sets -- no code execution / sandbox.

  Reward shaping:
    * exact (default): 1.0 iff ALL ground-truth calls matched and no extra call
      (or `[]` for the irrelevance/abstention examples), else 0.0.
    * partial: matched_calls / max(n_gt, n_pred) -- a denser signal.
    * format_bonus: small reward for emitting a parseable call list even when
      wrong (helps a base model discover the output format early).
  """

  system_message: str = (
      'You are a function-calling assistant. You are given a set of tools. '
      'Decide which tool(s) to call to answer the user, and output ONLY the '
      'call(s) as a Python list on a single line, e.g. '
      '[func_name(arg1=val1, arg2=val2)]. Call multiple tools by listing '
      'them: [f(a=1), g(b=2)]. If NO tool applies, output exactly [].'
  )
  partial_credit: bool = False
  format_bonus: float = 0.0
  # A few-shot exemplar block (inserted before the user turn) so a PT/base model
  # can pick up the output format AND learn to STOP after the call. Empty ->
  # zero-shot (base models tend to ramble past the answer and get truncated).
  few_shot: str = BFCL_FEW_SHOT

  def _format_tools(self, functions) -> str:
    """Renders the tool schemas as one-line signatures.

    Each tool becomes `- name(arg: type (required|optional), ...): description`.
    Part of the FIXED eval; the same rendering is used for the training reward.

    Args:
      functions: the example's tool schemas.

    Returns:
      The rendered tool block.
    """
    lines = []
    for f in functions:
      params = f.get('parameters', {}) or {}
      props = params.get('properties', {})
      if isinstance(props, str):
        try:
          props = json.loads(props)
        except (ValueError, TypeError):
          props = {}
      required = params.get('required', []) or []
      arg_strs = []
      for name, spec in props.items():
        t = spec.get('type', 'any') if isinstance(spec, dict) else 'any'
        req = 'required' if name in required else 'optional'
        arg_strs.append(f'{name}: {t} ({req})')
      lines.append(
          f"- {f['name']}({', '.join(arg_strs)}): {f.get('description', '')}"
      )
    return '\n'.join(lines)

  def get_prompt(self, example: Mapping[str, Any]) -> str:
    q = example['question']
    # `question` may be: a plain user-text string (ToolACE, serialized to keep
    # example fields flat/hashable for the RL pipeline), or the BFCL nested form
    # [[{role,content},...]].
    if isinstance(q, str):
      user_text = q
    else:
      turns = q[0] if q and isinstance(q[0], list) else q
      user_text = ' '.join(
          t['content'] for t in turns if t.get('role') == 'user'
      )
    functions = example['function']
    # `function` may be a JSON string (ToolACE serializes it so no nested dict
    # flows as a raw example field into the RL data pipeline).
    if isinstance(functions, str):
      functions = json.loads(functions)
    tools = self._format_tools(functions)
    # Layout: tool list, exemplar block, user turn. FIXED -- every reference
    # number was measured with this ordering; do not reorder.
    fs = f'{self.few_shot}\n' if self.few_shot else ''
    return f'Tools:\n{tools}\n\n{fs}User: {user_text}\nCalls: '

  def get_messages(
      self, example: Mapping[str, Any]
  ) -> Sequence[Mapping[str, Any]]:
    # The `Pretrain` lm_format only supports a single message, so we fold the
    # system instructions into one user turn (mirrors FewShotGSM8KEvaluation).
    prefix = f'{self.system_message}\n\n' if self.system_message else ''
    return [dict(role='user', content=prefix + self.get_prompt(example))]

  def evaluate(
      self, example: Mapping[str, Any], response: str
  ) -> Mapping[str, Any]:
    gt = example.get('ground_truth')
    # ground_truth may be a JSON string (ToolACE stores it serialized so the
    # nested dict/list arg values stay hashable through the data pipeline).
    if isinstance(gt, str):
      gt = json.loads(gt) if gt else None
    # Build name -> schema map for type-aware value checking (upstream parity).
    functions = example.get('function')
    if isinstance(functions, str):
      try:
        functions = json.loads(functions)
      except (ValueError, TypeError):
        functions = []
    schema_by_name = {
        f.get('name'): f for f in functions or [] if isinstance(f, dict)
    }
    calls = bfcl_parse_calls(response)
    parsed_ok = calls is not None
    fmt = float(parsed_ok)
    # Abstention / irrelevance examples: correct iff an empty call list.
    if gt is None:
      # Upstream (relevance_file_runner) scores abstention CORRECT when it
      # makes no function call -- either a plain-text refusal (no parse) or
      # an explicit empty list `[]`. Both count.
      correct = (not parsed_ok) or (len(calls) == 0)
      reward = float(correct)
      if not correct and self.format_bonus:
        reward = max(reward, self.format_bonus * fmt)
      return {'correct': int(correct), 'reward': reward, 'format': fmt}

    if not parsed_ok:
      return {'correct': 0, 'reward': 0.0, 'format': 0.0}

    gt_list = list(gt)
    used = [False] * len(calls)
    matched = 0
    for gtc in gt_list:
      for i, (pn, pa) in enumerate(calls):
        if not used[i] and _single_call_match(pn, pa, gtc, schema_by_name):
          used[i] = True
          matched += 1
          break
    full = (matched == len(gt_list)) and (len(calls) == len(gt_list))
    if self.partial_credit:
      denom = max(len(gt_list), len(calls), 1)
      reward = matched / denom
    else:
      reward = float(full)
    if not full and self.format_bonus:
      reward = max(reward, self.format_bonus * fmt)
    return {
        'correct': int(full),
        'reward': float(reward),
        'format': fmt,
        'matched': matched,
        'n_gt': len(gt_list),
        'n_pred': len(calls),
    }
