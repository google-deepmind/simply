# Running the research-bench tasks on Google Cloud

Everything here is run from the **repository root** with
`tasks/research_bench/launch/launch_gcp.py`, which replaces the internal
sweep launcher: it stages your working tree, creates Cloud TPU VMs, runs one
seed per VM, and leaves a submission-shaped directory in GCS.

```
gs://<bucket>/<experiment_name>/launch_manifest.json
gs://<bucket>/<experiment_name>/seed_<seed>/final_result.json   <- the metric
gs://<bucket>/<experiment_name>/seed_<seed>/status.json         <- COMPLETED/FAILED
gs://<bucket>/<experiment_name>/seed_<seed>/log.txt             <- stdout+stderr
gs://<bucket>/<experiment_name>/seed_<seed>/tb_log/...
gs://<bucket>/<experiment_name>/seed_<seed>/experiment_config.json
```

That directory *is* the submission: point the validator at it (§6).

For generic Simply-on-GCP background (VPC, NAT, multi-host, GPUs, profiling)
see [`docs/gcloud.md`](../../docs/gcloud.md); this file only covers the
benchmark flow.

---

## 1. One-time project setup

```bash
PROJECT=my-project
BUCKET=gs://my-project-rbv3          # co-locate with the TPU region!
ZONE=us-central1-b

gcloud services enable tpu.googleapis.com --project=$PROJECT
gcloud storage buckets create $BUCKET --location=us-central1 --project=$PROJECT

# The TPU VM's service account needs to read code/assets and write results.
SA="$(gcloud iam service-accounts list --project=$PROJECT \
      --filter='email:compute@developer.gserviceaccount.com' \
      --format='value(email)')"
gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:$SA" --role=roles/storage.admin
```

* **Quota**: `TPUv6e` (or `TPUv5litepod`) chips in your zone, under
  IAM & Admin > Quotas. The 3-seed sweep of a `v6e-4` task with
  `--num-workers=3` needs 12 chips at once; `--num-workers=1` needs 4.
* **Firewall**: nothing to open for `--transport=gcs`. For `--transport=ssh`
  you need `tcp:22` inbound (`gcloud compute firewall-rules create allow-ssh
  --network=default --allow=tcp:22,icmp --project=$PROJECT`), or IAP
  (`35.235.240.0/20 tcp:22`).
* **Network**: VMs need outbound internet for `pip install` (external IP, the
  default, or Cloud NAT).

Check the launcher can see your project and that SSH works from your machine:

```bash
python -m tasks.research_bench.launch.launch_gcp probe-ssh \
    --project=$PROJECT --zone=$ZONE
```

## 2. Assets

Each task declares what it needs; stage it once into the cache and mirror it
to GCS (see [`setup/`](setup/)):

```bash
pip install ".[assets]"
python -m tasks.research_bench.setup.prepare_assets task pretrain_bpb_byte \
    --gcs-bucket $BUCKET          # -> $BUCKET/assets/{models,datasets,vocabs}
```

Pass `--assets=$BUCKET/assets` to the launcher; the VM mirrors it to local disk
and exports `SIMPLY_MODELS` / `SIMPLY_DATASETS` / `SIMPLY_VOCABS`.

The full cache is ~90 GB, so mirror only what the task reads:

```bash
--assets=$BUCKET/assets --assets-include=datasets/c4_bin,vocabs      # pretraining
--assets=$BUCKET/assets --assets-include=models/GEMMA-3.0-1B-PT-ORBAX,datasets,vocabs
```

`--assets-mode=direct` skips the copy and points the env vars at `gs://`
instead — fine for vocabs, but the C4 `.bin/.idx` shards and the 1.1 GiB
LiveCodeBench jsonl are memory-mapped and must be local.

### VM software beyond the base image

The launcher installs this for you (`--pip-extras`, `--apt`, defaults per
task); the list matters only if you run a task by hand on a VM:

| task family | needs |
|---|---|
| pretraining, RL, porting | `pip install ".[tpu,gcloud,math-eval]"` — the boxed-answer reward imports `sympy` |
| `sampling_lcb`, `decode_efficiency_vf` | `pip install ".[tpu,gcloud,serving,math-eval]"` **and** `python setup/gen_protos.py` — `simply/serving/*_pb2.py` are generated, not checked in, and `simply.eval.page_decode_eval` imports them at module load |
| `sampling_lcb` | `sudo apt-get install -y bubblewrap` — the grader runs model-written Python in it; the run logs its actual isolation as `[code_exec] sandbox=...` |

## 3. Dry run first — it is free

```bash
python -m tasks.research_bench.launch.launch_gcp \
    --task=pretrain_bpb_byte --experiment_name=bpb_byte_v1 \
    --bucket=$BUCKET --seeds=42,43,44 --dry-run
```

Prints the manifest and the exact per-seed command line and touches nothing.
Do this whenever you change a config or a flag.

## 4. Launch a baseline

`--task` selects the entry point, the accelerator, the task's fixed config
overrides and (for the eval-only tasks) the decoding protocol — see
[`launch/task_defaults.py`](launch/task_defaults.py). Everything is
overridable.

```bash
COMMON="--bucket=$BUCKET --zone=$ZONE --project=$PROJECT --assets=$BUCKET/assets"

# Pretraining tasks (pretrain_bpb_v32k, pretrain_bpb_byte, pretrain_optimizer_ttt)
python -m tasks.research_bench.launch.launch_gcp $COMMON \
    --task=pretrain_bpb_byte --experiment_name=bpb_byte_v1 \
    --seeds=42,43,44 --num-workers=3 --assets-include=datasets/c4_bin,vocabs

# RL tasks (rl_gemma3_1b, rl_bfcl_*, rl_qwen2p5_math_1p5b)
python -m tasks.research_bench.launch.launch_gcp $COMMON \
    --task=rl_gemma3_1b --experiment_name=rl_gemma3_v1 --seeds=42,43,44

# Eval-only tasks: no training, a released checkpoint is sampled.
# sampling_lcb runs on v6e-4; decode_efficiency_vf is FIXED at v6e-8 because
# its metric is wall-clock.
python -m tasks.research_bench.launch.launch_gcp $COMMON \
    --task=sampling_lcb --experiment_name=lcb_v1 --seeds=42,43,44

# Porting tasks: the ported model's run, then the reference port run
python -m tasks.research_bench.launch.launch_gcp $COMMON \
    --task=port_falcon_h1_0p5b --experiment_name=falcon_v1 --seeds=42,43,44
python -m tasks.research_bench.launch.launch_gcp $COMMON \
    --task=port_falcon_h1_0p5b --experiment_name=falcon_v1/port_run \
    --seeds=42,43,44 --extra-flag=port_run=true
```

Useful flags:

| flag | what it does |
|---|---|
| `--num-workers=N` | N TPU VMs, seeds spread over them. `1` = all seeds one after another on one VM (cheapest, 3x slower). |
| `--spot` | spot capacity: ~50 % cheaper, preemptible (§7). |
| `--keep-vms` | don't delete the VMs afterwards (debugging, or a fast second launch). |
| `--vms=a,b` | run on TPU VMs you already created (or that a previous sweep kept); the launcher never deletes those, and recreates one under the same name if it dies. |
| `--zone=a,b,c` | rotate through zones when one is out of capacity. |
| `--spot-fallback` | try on demand across every `--zone` first, then retry on spot. |
| `--apt=pkg,...` | extra system packages on the VM (default: per task). |
| `--extra-flag=k=v` | appended as `--k=v` to the remote binary. Repeatable; overrides a task default with the same name. |
| `--assets-include=a,b` | mirror only these subpaths of the asset cache. |
| `--verify-assets` | crc32c every mirrored file against the staged manifest before running (~1 min per 80 GB). **Use it for any checkpoint-loading task.** |
| `--config-overlay='{"num_train_steps": 100}'` | JSON merged into `--config_overlay` (the seeds are added to it). Handy for a short test run. |
| `--entry-module=...` | run a different module (e.g. your own `main`). |
| `--force` | re-run seeds that already have a `final_result.json`. |

**Restartability.** Re-running the *same* command resumes: finished seeds are
skipped, live VMs are reused, unfinished seeds are re-run. The manifest keeps
the first `launched_at` (the validator's anti-replay check depends on it) and
appends the new invocation to `launches[]`.

## 5. Watch it

```bash
L="python -m tasks.research_bench.launch.launch_gcp"
$L status  --bucket=$BUCKET --experiment_name=bpb_byte_v1
$L logs    --bucket=$BUCKET --experiment_name=bpb_byte_v1 --seed=42 --follow
$L collect --bucket=$BUCKET --experiment_name=bpb_byte_v1          # metrics + mean
$L collect --bucket=$BUCKET --experiment_name=bpb_byte_v1 --json
tensorboard --logdir $BUCKET/bpb_byte_v1
```

`log.txt` is refreshed every ~20 s while the run is alive, so `logs --follow`
works even when you cannot SSH to the VM.

## 6. Validate the submission

```bash
python -m tasks.research_bench.validator.run_validator \
    --task=pretrain_bpb_byte --experiment_dir=$BUCKET/bpb_byte_v1
```

Exit 0 means the submission is well-formed and scoreable. The launcher writes
everything the validator needs: `task`, `seeds`, `git_commit`, `git_dirty`,
`launched_at`, `tpu_type` in `launch_manifest.json`, and a per-seed
`status.json`.

## 7. Preemption and capacity

Spot TPUs are reclaimed without warning (observed lifetimes in one project:
14–40 min, median ~23). The launcher notices within ~2 min that a VM went
`PREEMPTED`/away, deletes it, recreates it and re-runs that seed from scratch
(up to `--max-preemptions`, default 5). Two consequences:

* **Only use `--spot` for runs shorter than the typical spot lifetime**, or
  for configs that checkpoint to `--experiment_dir` and resume (the seed
  restarts from the last checkpoint in GCS, not from step 0).
* On-demand capacity for large slices is scarce. If `create` keeps returning
  *"There is no more capacity in the zone"*, the launcher retries
  (`--create-retries`, default 20 × 60 s). Try another zone, a smaller
  `--tpu-type`, or `--spot`; v5e (`v5litepod-*`) is usually far easier to get
  than v6e.

```bash
# tear down anything left behind (also run automatically unless --keep-vms)
$L teardown --bucket=$BUCKET --experiment_name=bpb_byte_v1
gcloud compute tpus tpu-vm list --zone=$ZONE --project=$PROJECT
gcloud compute tpus queued-resources list --zone=$ZONE --project=$PROJECT  # these outlive nodes
```

## 8. Cost and time

Approximate list prices per **chip**-hour (check
<https://cloud.google.com/tpu/pricing>; spot is dynamic):

| accelerator | chips | on-demand $/h | spot $/h | typical use here |
|---|---|---|---|---|
| `v5litepod-1` | 1 | ~1.20 | ~0.60 | smoke tests |
| `v6e-1` | 1 | ~2.70 | ~1.35 | smoke tests |
| `v6e-4` | 4 | ~10.8 | ~5.4 | 10 of the 11 tasks |
| `v6e-8` | 8 | ~21.6 | ~10.8 | `decode_efficiency_vf` only |

Reference wall-clock per seed on a 4-chip host (from the internal bundle; also
the source of the default `--timeout-min`, which is 4x these):

| task | one seed |
|---|---|
| `pretrain_bpb_byte` | 12-18 min |
| `pretrain_bpb_v32k` | 15-20 min |
| `pretrain_optimizer_ttt` | 25-35 min |
| `rl_gemma3_1b` | ~20 min |
| `rl_bfcl_gemma3_1b` | 20-25 min |
| `rl_bfcl_qwen3_0p6b` | 40-45 min |
| `rl_qwen2p5_math_1p5b` | ~2 h |
| `sampling_lcb` | 25-30 min |

Measured here on **v6e-4** (europe-west4-a, warm VM), which is what you should
actually budget from:

| task | measured | note |
|---|---|---|
| the three pretraining sweeps | **4-6 min per seed** | several times faster than the reference above |
| `sampling_lcb` | **20.8 s per example** | x167 problems = **~1 h per seed**, not 25-30 min |
| `rl_gemma3_1b` | **~8 min per held-out eval** (full 1319-example GSM8K test) | the RL tasks are **eval-dominated**: at `validation_eval_interval=50` the evals, not the training steps, set the wall-clock |

Add ~6 min per VM for first-boot setup (apt + `pip install -e .`), and the
asset mirror (seconds for vocabs, ~1 min for the 25.6 GB C4 repack, longer for
a multi-GB checkpoint). A 3-seed sweep costs 3x one seed whether you run it on
3 VMs (fast) or 1 VM (`--num-workers=1`, 3x longer) — the chip-hours are the
same.

**Boot disk.** A TPU VM's boot disk is ~97 GB and is **not configurable**
(`tpu-vm create` has no size flag). It holds the venv, the code and the
mirrored assets, so mirror selectively: `decode_efficiency_vf`'s
Qwen3-30B-A3B checkpoint alone is 42.5 GiB, and the whole asset cache is
~90 GB and will not fit.

```bash
--assets-include=models/Qwen3-30B-A3B-Thinking-2507,vocabs,datasets/aime
```

## 9. If you cannot SSH to a TPU VM (`--transport=gcs`)

Some networks (corp egress policies, IAP-less VPCs) block
`gcloud compute tpus tpu-vm ssh` — the symptom is a hang or
`websocket: close 4003: failed to connect to backend`. Check with
`launch_gcp probe-ssh`.

`--transport=gcs` removes SSH from the path entirely:

```bash
$L --task=pretrain_bpb_byte --experiment_name=bpb_byte_v1 \
   --bucket=$BUCKET --zone=$ZONE --transport=gcs ...
```

The VM's `startup-script` installs a small agent
([`launch/agent.py`](launch/agent.py)) that polls
`gs://<bucket>/_simply_ctl/<vm>/cmd/` for job scripts, runs them as root, and
streams their logs back to GCS. Everything else is identical. Notes:

* The agent authenticates with the VM's own service account, so the VM needs
  `--scopes=cloud-platform` (the launcher always sets this) and write access
  to the bucket.
* A job id runs **once per VM, ever** — the launcher always submits a fresh id.
* `--ctl-prefix` moves the job queue if `_simply_ctl` clashes with something.
* Debugging without SSH: read `gs://<bucket>/_simply_ctl/<vm>/heartbeat.txt`
  (agent liveness) and `.../out/<job-id>.log`, or submit your own script:
  `echo 'journalctl -u simply-agent | tail -50' | gcloud storage cp - \
   gs://<bucket>/_simply_ctl/<vm>/cmd/peek1.sh` then read
  `.../out/peek1.log`.

## 9b. What is not verified

Be aware of the edges of what has actually been run:

* **`--transport=ssh` has never been exercised end to end.** It is the
  documented default because it is the right path for a normal user, but every
  run behind the numbers in this suite used `--transport=gcs`, because SSH to
  GCP is blocked from the machine they were run on. Run `probe-ssh` first.
* **Multi-host slices (v6e-16 and up) are not supported.** The ssh transport
  would run the job via `--worker=all`; the gcs transport installs one agent
  per VM and has no notion of a multi-host slice. Every task in this suite is
  single-host (v6e-1/4/8), which is what is verified.
* **`decode_efficiency_vf` has never completed a run here, and its
  `avg_generation_time` anchors are still the internal ones.** Everything up
  to decoding works -- the 42.5 GiB Qwen3-30B-A3B checkpoint shards and loads
  on a real `1,1,2,4` mesh, sparse MoE with expert parallelism compiles -- but
  the task needs ~30 min of setup (apt, pip, the 45.6 GB asset mirror, load and
  compile) before the first token, and the **v6e-8 spot nodes we could get
  lived 6-7 minutes**. On spot it therefore cannot finish, however good the
  retry logic is. It needs **on-demand v6e-8 or a reservation**; budget ~40 min
  of capacity hunting on top.

## 10. Troubleshooting

| symptom | fix |
|---|---|
| `OSError: [Errno 24] Too many open files` from grain | the launcher raises `ulimit -n` in the job; if you run by hand, do the same. |
| the job sits in `Waiting for cache lock ... (unattended-upgr)` | normal for the first ~15 min of a VM's life; apt waits it out. |
| `_.gstmp: No such file or directory` while mirroring assets | two jobs are mirroring into the same directory — you have duplicate jobs on one VM. |
| `Truncated Zstd-compressed stream` from tensorstore when loading a checkpoint | the mirrored file has the right size and the wrong bytes (a resumed sliced download). The launcher disables sliced downloads for the mirror; add `--verify-assets` to catch it before the TPU is touched, and note that `rsync` will never re-fetch such a file because size and mtime match. |
| `gcloud` auth rejected by an org policy | `--adc-token` (mints and refreshes `CLOUDSDK_AUTH_ACCESS_TOKEN` from application-default credentials). |
| seed says `exit=0 but no final_result.json` | the run ended early; read `log.txt`. Usually a config that never reaches its last step. |
| `Permission denied` writing to the bucket | the TPU service account is missing `roles/storage.admin` (§1). |
| `no more capacity in the zone` | §7. |
| pip install fails on the VM | no outbound internet: give the VM an external IP or set up Cloud NAT (`docs/gcloud.md`). |
| a stale VM still costs money | `$L teardown`, and check `queued-resources list`. |
