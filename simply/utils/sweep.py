"""Hyperparameter sweep combinator library / DSL.

A sweep is defined as a list of nested dictionaries, which can then be overlaid
onto a config dataclass.

## Reference

The DSL (evaluated with `eval_sweep`) is a subset of Python with the following
primitives:

**Built-ins**:
- `dict`, `range`, `np`
**Primitives**:
- `param(name, *values)` / `p(name, *values)`: Sweep a parameter over a list of
values. `name` supports nesting via `.`
- `default`: No changes (useful for chaining as a baseline).
**Combinators**:
- `prod(*sweeps)`: Cartesian product of sweeps.
- `chain(*sweeps)`: Sum of sweeps.
- `zipit(*sweeps)`: Zip together sweeps.
- `prefix(path, *sweeps, **fixed)`: Prefix a sweep with a path.
- `subsample(n, sweep)`: Randomly subsample a sweep.

The sweep definition must be a **single expression**.

## Examples

```python
prod(
    param("optimizer.lr", 1e-3, 1e-4),
    param("data.batch_size", range(32, 128, 32)),
)
```

```python
chain(
  default,
  prod(
    prefix("optimizer",
      zipit(
        p("lr", 2e-3, 1e-3),
        p("momentum", [0.5, 0.8]),
      )
    ),
    param("data.batch_size", [32, 64])
  )
)
```

```python
chain(
  default,
  subsample(8, # randomly subsample from a large product space.
    prod(
      prefix("optimizer",
          param("lr", 2e-3, 1e-3, 3e-4, 1e-4),
          param("momentum", [0.2, 0.5, 0.8]),
          param("name", "adam", "muon")
      ),
      param("data.batch_size", [16, 32, 64, 128, 256])
    )
  )
)
```
"""

from collections.abc import Callable, Iterable
import dataclasses
import functools
import itertools
import os
from typing import Any, TypeVar
import numpy as np

HParams = dict[str, Any]
Sweep = list[HParams]
T = TypeVar("T")


def _nest(hparam: HParams, prefix: str = "") -> HParams:
  """Converts a flat dict with dotted keys into a nested dict."""
  out = dict()
  if prefix and not prefix.endswith("."):
    prefix += "."
  for k, v in hparam.items():
    d = out
    *subkeys, key = (prefix + k).split(".")
    if not subkeys:
      out[key] = v
    else:
      for subkey in subkeys:
        d = d.setdefault(subkey, {})
      d[key] = v
  return out


def _merge(hparams: Iterable[HParams] | HParams):
  """If hparams is a seq of dicts, merge them into a single dict."""
  if isinstance(hparams, dict):
    return hparams
  return functools.reduce(dict.__or__, hparams, {})


def sweep_combinator(
    fn: Callable[..., Iterable[Sweep | HParams]],
) -> Callable[..., Sweep]:
  """Marks a function as a sweep combinator, which returns a list of flat hparam dicts."""

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    out = fn(*args, **kwargs)
    return [_merge(x) for x in out]

  return wrapper


@sweep_combinator
def param(name, *values):
  """Sweep a parameter over a list of values."""
  return [{name: v} for v in values]


@sweep_combinator
def prod(*sweeps: Sweep):
  """Product (cross-product) a number of sweeps."""
  return itertools.product(*sweeps)


@sweep_combinator
def chain(*sweeps: Sweep):
  """Chain (sum) a number of sweeps."""
  return itertools.chain.from_iterable(sweeps)


@sweep_combinator
def zipit(*sweeps: Sweep):
  """Zip together a number of sweeps."""
  return zip(*sweeps)


# Underscore to avoid name collision with prefix variables.
@sweep_combinator
def _prefix(path: str, *sweeps: Sweep, **fixed):
  """Prefix a sweep with a path."""
  if fixed:
    sweeps = [[fixed], *sweeps]  # pyrefly: ignore[bad-assignment]

  return map(lambda x: {path: x}, prod(*sweeps))


@sweep_combinator
def subsample(n: int, sweep: Sweep, seed=299792458):
  """A random subsample of a sweep."""
  rng = np.random.default_rng(seed)

  sweep = list(sweep)
  n = min(len(sweep), n)
  if n:
    for i in rng.choice(len(sweep), size=n, replace=False):
      yield sweep[i]


def overlay_from(root: T, updates, *, strict=True) -> T:
  """Overlay a sparse nested tree of updates onto a root dataclass.

  Args:
    root: The root dataclass to overlay onto.
    updates: A dictionary of updates to overlay onto root.
    strict: If False, filter out invalid fields rather than raising an error.

  Returns:
    An updated dataclass instance.
  """

  def _update(old, new):
    if isinstance(new, dict):
      if old is None:
        return old
      return overlay_from(old, new, strict=strict)
    else:
      return new

  new_fields = set(updates.keys())
  old_fields = {f.name for f in dataclasses.fields(root)}  # pyrefly: ignore[bad-argument-type]
  if strict and not new_fields.issubset(old_fields):
    raise ValueError(
        f"New fields {new_fields - old_fields} not found in root"
        f" {old_fields}; root type: {type(root).__name__}"
    )

  new_vals = {
      k: _update(getattr(root, k), updates[k]) for k in old_fields & new_fields
  }
  return dataclasses.replace(root, **new_vals)  # pyrefly: ignore[bad-specialization]


def eval_sweep(
    s: str,
    extra_symbols: dict[str, Any] | None = None,
) -> Sweep:
  """Parse a sweep string.

  Args:
    s: The sweep string to parse.
    extra_symbols: Extra symbols to allow in the sweep string.

  Returns:
    A list of hyperparameter dictionaries.
  """
  if not s:
    return [{}]

  s = s.strip()

  # If we pass in a file, read it.
  if os.path.exists(s):
    with open(s, "r") as f:
      s = f.read()

  allowed_symbols = dict(
      # Python builtins
      np=np,
      range=range,
      dict=dict,
      default=[{}],
      param=param,
      p=param,  # shorthand
      prod=prod,
      chain=chain,
      prefix=_prefix,
      zipit=zipit,
      subsample=subsample,
      **(extra_symbols or {}),
  )
  # Python doesn't like using these as args to dict()
  allowed_symbols |= {"True": True, "False": False, "None": None}

  # semiliteral_eval lacks support for *args and binops, so we use
  # raw eval(). we pass in globals explicitly to control what's available.
  sweep = eval(s, allowed_symbols)  # pylint: disable=eval-used
  if isinstance(sweep, tuple):
    sweep = chain(*sweep)
  elif isinstance(sweep, dict):
    sweep = [sweep]
  elif callable(sweep):
    # Registry functions resolve to callables, so we need to call it.
    sweep = sweep()
  return sweep


def overlay(default: T, updates: HParams, prefix: str = "") -> T:
  return overlay_from(default, _nest(updates, prefix))
