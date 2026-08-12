r"""Serialize the official SWE-Bench Pro dataset into a single JSON.

Reads the official ScaleAI release laid out under `--src_dir`:

  SWE-bench_Pro/data/test-00000-of-00001.parquet    (731 tasks, 16 columns)
  SWE-bench_Pro-os/run_scripts/<instance_id>/{run_script.sh,parser.py}

and writes a single JSON list (one object per instance) containing ONLY the
original data:

  * every official parquet column, verbatim, under its original name;
  * the two per-instance run-script files, verbatim, under their ORIGINAL
    names: `run_script.sh` and `parser.py` (nested under `run_scripts`).

No derived / internal-env fields are written (no sandbox image id, no wrapped
prompt, no test_cmd, ...). Those are derived on the fly by
`simply.data_lib.SweBenchProSource` at load time. Consolidating into one JSON
avoids the ~1.5k per-file remote reads that make direct-from-source loading
slow.

Run:
  python -m simply.tools.serialize_swe_bench_pro \
      --src_dir=$SRC_DIR --out=$OUT_JSON_PATH
"""

import concurrent.futures
import json

from absl import app
from absl import flags
from absl import logging
from etils import epath
import pyarrow as pa
import pyarrow.parquet as pq

_SRC_DIR = flags.DEFINE_string(
    'src_dir',
    None,
    'Root of the official SWE-Bench Pro download (contains SWE-bench_Pro/ and'
    ' SWE-bench_Pro-os/).',
)
_OUT = flags.DEFINE_string('out', None, 'Output JSON path.')
_INDENT = flags.DEFINE_bool(
    'indent', False, 'Pretty-print the JSON output (larger file).'
)
_MAX_WORKERS = flags.DEFINE_integer(
    'max_workers', 64, 'Parallelism for reading per-instance run scripts.'
)

# Run-script files kept, under their ORIGINAL names.
_RUN_SCRIPT_FILES = ('run_script.sh', 'parser.py')


def _read_text(path: str | epath.PathLike) -> str:
  """Reads a remote/local text file; returns '' (logged) on failure."""
  try:
    return epath.Path(path).read_text(encoding='utf-8')
  except Exception as e:  # pylint: disable=broad-except
    logging.warning('read failed for %s: %r', path, e)
    return ''


def _read_run_scripts(
    run_scripts_dir: epath.Path, instance_id: str
) -> dict[str, str]:
  """Returns {original_filename: content} for one instance's run scripts."""
  out = {}
  for fname in _RUN_SCRIPT_FILES:
    text = _read_text(run_scripts_dir / instance_id / fname)
    if not text:
      logging.warning('empty/missing %s for instance %s', fname, instance_id)
    out[fname] = text
  return out


def main(argv):
  del argv

  src_dir = epath.Path(_SRC_DIR.value)  # pyrefly: ignore[bad-argument-type]
  parquet_path = src_dir / 'SWE-bench_Pro/data/test-00000-of-00001.parquet'
  run_scripts_dir = src_dir / 'SWE-bench_Pro-os/run_scripts'

  # Read the parquet (fast) into a list of row dicts. pyarrow can't open a
  # remote path directly, so read bytes via epath and wrap in a BufferReader.
  with parquet_path.open('rb') as f:
    table = pq.read_table(pa.BufferReader(f.read()))
  rows = table.to_pylist()
  logging.info('loaded %d rows from %s', len(rows), parquet_path)

  # Normalize any bytes -> str (official columns are all text/list-of-text).
  for p in rows:
    for k, v in list(p.items()):
      if isinstance(v, bytes):
        p[k] = v.decode('utf-8')

  instance_ids = [p['instance_id'] for p in rows]

  # Read all per-instance run scripts in parallel (the slow part when done
  # one-by-one at load time: ~1.5k remote reads).
  logging.info(
      'reading run scripts for %d instances with %d workers...',
      len(instance_ids),
      _MAX_WORKERS.value,
  )
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=_MAX_WORKERS.value
  ) as ex:
    run_scripts_list = list(
        ex.map(
            lambda iid: _read_run_scripts(run_scripts_dir, iid), instance_ids
        )
    )

  out_rows = []
  for p, run_scripts in zip(rows, run_scripts_list):
    item = dict(p)  # every official column, verbatim.
    item['run_scripts'] = run_scripts  # {run_script.sh: ..., parser.py: ...}
    out_rows.append(item)

  out_path = epath.Path(_OUT.value)  # pyrefly: ignore[bad-argument-type]
  logging.info('writing %d rows to %s', len(out_rows), out_path)
  with out_path.open('w') as f:
    json.dump(
        out_rows,
        f,
        indent=2 if _INDENT.value else None,
        ensure_ascii=False,
    )
  logging.info('done')


if __name__ == '__main__':
  app.run(main)
