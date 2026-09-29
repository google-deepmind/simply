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


"""On-VM agent: runs job scripts dropped in GCS and streams their logs back.

Installed by the TPU VM startup script (see `remote.py`) and used by
`--transport=gcs`, the SSH-less path. Talks to the GCS JSON API with the VM's
own service-account token, so it needs no pip install at boot.

Layout under gs://<ctl-bucket>/<ctl-prefix>/<node>/ :
  cmd/<id>.sh      job script -- the agent runs it, as root, once ever
  out/<id>.log     stdout+stderr, mirrored while the job runs
  out/<id>.status  "running", then "exit=<code>"
  heartbeat.txt    {"t": ..., "running": [...], "done": [...]} every ~5 s
"""

import json
import os
import signal
import socket
import subprocess
import time
import urllib.parse
import urllib.request

MD = 'http://metadata.google.internal/computeMetadata/v1/'
STATE = '/var/lib/simply-agent'
POLL_SEC = 5
MAX_LOG = 512 * 1024  # only a log's tail is mirrored ...
BIG_LOG_EVERY = 30  # ... and a big log is mirrored less often.


def md(path, default=''):
  req = urllib.request.Request(MD + path, headers={'Metadata-Flavor': 'Google'})
  try:
    return urllib.request.urlopen(req, timeout=10).read().decode()
  except Exception:  # pylint: disable=broad-except
    return default


def _api(url, method='GET', data=None, ctype=None):
  token = json.loads(md('instance/service-accounts/default/token'))
  req = urllib.request.Request(url, data=data, method=method)
  req.add_header('Authorization', 'Bearer ' + token['access_token'])
  if ctype:
    req.add_header('Content-Type', ctype)
  return urllib.request.urlopen(req, timeout=120).read()


def gcs_put(bucket, obj, data):
  if isinstance(data, str):
    data = data.encode()
  url = (f'https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o'
         f'?uploadType=media&name={urllib.parse.quote(obj, safe="")}')
  _api(url, 'POST', data, 'application/octet-stream')


def gcs_get(bucket, obj):
  url = (f'https://storage.googleapis.com/storage/v1/b/{bucket}/o/'
         f'{urllib.parse.quote(obj, safe="")}?alt=media')
  return _api(url)


def gcs_list(bucket, prefix):
  url = (f'https://storage.googleapis.com/storage/v1/b/{bucket}/o'
         f'?prefix={urllib.parse.quote(prefix, safe="")}&maxResults=1000')
  return [i['name'] for i in json.loads(_api(url)).get('items', [])]


def node_id():
  """Control-plane identity: TPU node names are only unique within a zone."""
  name = md('instance/attributes/ctl-node')
  if not name:
    for line in md('instance/attributes/tpu-env').splitlines():
      if line.startswith('NODE_NAME:'):
        name = line.split(':', 1)[1].strip().strip('\'"')
  return name or socket.gethostname()


def _cycle_timeout(signum, frame):
  del signum, frame
  raise TimeoutError('agent cycle exceeded its deadline')


class Job:
  """A job script running as a detached root process."""

  def __init__(self, bucket, base, jid, script):
    self.bucket, self.base, self.jid = bucket, base, jid
    os.makedirs(f'{STATE}/jobs', exist_ok=True)
    self.log_path = f'{STATE}/jobs/{jid}.log'
    path = f'{STATE}/jobs/{jid}.sh'
    with open(path, 'wb') as f:
      f.write(script)
    self.log = open(self.log_path, 'wb')
    self.proc = subprocess.Popen(
        ['/bin/bash', path], stdout=self.log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, cwd='/tmp', start_new_session=True,
        env={**os.environ, 'HOME': '/root', 'JOB_ID': jid})
    self.sent = -1
    self.next_pump = 0.0
    gcs_put(bucket, f'{base}/out/{jid}.status', 'running\n')

  def pump(self, final=False):
    size = os.path.getsize(self.log_path)
    now = time.time()
    if size == self.sent and not final:
      return
    if size > MAX_LOG and not final and now < self.next_pump:
      return
    with open(self.log_path, 'rb') as f:
      if size > MAX_LOG:
        f.seek(size - MAX_LOG)
        data = b'[... %d earlier bytes omitted ...]\n' % (size - MAX_LOG)
        data += f.read()
      else:
        data = f.read()
    gcs_put(self.bucket, f'{self.base}/out/{self.jid}.log', data)
    self.sent = size
    self.next_pump = now + BIG_LOG_EVERY

  def poll(self):
    """True once the process has exited (and its final log is uploaded)."""
    rc = self.proc.poll()
    if rc is None:
      self.pump()
      return False
    self.log.flush()
    self.pump(final=True)
    gcs_put(self.bucket, f'{self.base}/out/{self.jid}.status', f'exit={rc}\n')
    return True


def main():
  bucket = md('instance/attributes/ctl-bucket').strip()
  bucket = bucket.replace('gs://', '').strip('/')
  prefix = (md('instance/attributes/ctl-prefix') or '_ctl').strip().strip('/')
  node = node_id()
  base = f'{prefix}/{node}'
  os.makedirs(f'{STATE}/jobs', exist_ok=True)
  # A job id runs once per VM, ever -- including across an agent restart.
  seen = {f[:-3] for f in os.listdir(f'{STATE}/jobs') if f.endswith('.sh')}

  gcs_put(bucket, f'{base}/boot.json', json.dumps({
      'node': node,
      'hostname': socket.gethostname(),
      'boot': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
      'accelerator': md('instance/attributes/accelerator-type'),
      'tpu_env': md('instance/attributes/tpu-env'),
  }, indent=2) + '\n')

  running = {}
  signal.signal(signal.SIGALRM, _cycle_timeout)
  while True:
    try:
      signal.alarm(180)  # a wedged GCS call must not freeze the agent forever
      for name in gcs_list(bucket, f'{base}/cmd/'):
        jid = os.path.basename(name)[:-3]
        if not name.endswith('.sh') or jid in seen:
          continue
        seen.add(jid)
        running[jid] = Job(bucket, base, jid, gcs_get(bucket, name))
      for jid in [j for j, job in running.items() if job.poll()]:
        del running[jid]
      gcs_put(bucket, f'{base}/heartbeat.txt', json.dumps({
          't': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
          'running': sorted(running),
          'done': sorted(seen - set(running)),
      }) + '\n')
    except Exception as e:  # pylint: disable=broad-except
      with open(f'{STATE}/agent.err', 'a') as f:
        f.write(f'{time.strftime("%FT%T")} {type(e).__name__}: {e}\n')
    finally:
      signal.alarm(0)
      open(f'{STATE}/alive', 'w').close()  # watched by the systemd watchdog
    time.sleep(POLL_SEC)


if __name__ == '__main__':
  main()
