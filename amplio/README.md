# Amplio: A Lightweight Agent Harness for Robust and Long-Horizon Runs

* **Lightweight**: Multi-agent framework based on a simple step model that provides natural crash-resume capability, and agent / user / env message coordination. Agents work through a small set of generic tools: shell, file edit, sub-agent spawn, and inter-agent coordination.
* **Robust**: DB-first persistence; crashing at any point during the agentic loop can be robustly resumed.
  * The agentic loop is guaranteed to be consistent, failures in the tool / environment are not handled by the harness, but reported to the agent.
  * Sub-agent session trees are automatically resumed upon server recovery.
* **Autonomous**: Each run can be fully driven by the autonomous agent, or cooperatively driven with user interactions.

## Installation

Amplio is a single binary with an embedded frontend.



**Build from source**: Needs Go (see `go.mod` for the version) and Node.js 22+ (try
[fnm](https://github.com/Schniz/fnm)). Then run `make build` in the root folder.

## Initial Setup

Amplio primarily needs two things:

1. **A data directory**: Amplio stores all agent runs in a local SQLite
   database under a data directory (`--data-dir` > `$AMPLIO_DATA_DIR` >
   `~/.amplio`). It also holds `config.toml`, per-run artifacts, blobs, and
   logs. No external database to provision.
2. **An LLM provider**: Amplio currently uses Vertex AI (via ADC) for Claude
   and Gemini, the Gemini Developer API (via API key), or an out-of-process
   `subprocess` bridge for backends that can't be linked into the binary.

### Configuration

Settings live in `<data-dir>/config.toml`. A minimal config:

```toml
# LLMs for system tasks such as auto critic. REQUIRED. Can also be
# specified via env vars AMPLIO_SYSTEM_LLM_HQ and AMPLIO_SYSTEM_LLM_FAST
system_llm_hq   = "vertex-claude:claude-opus-4-8"
system_llm_fast = "vertex-claude:claude-sonnet-4-6"

# If not set, recall (skills and knowledge) sub-system will be disabled.
# Can also be specified via env var AMPLIO_EMBED_MODEL
embed_model     = "vertex:text-embedding-005"

[run]
# list of models available to select from the new-run form
llms = [
  "vertex-claude:claude-opus-4-8",
  "vertex-gemini:gemini-3.5-flash",
]
```

For the Vertex AI providers, point ADC at your GCP project in `~/.bashrc`:

```bash
export VERTEXAI_PROJECT=<your-gcp-project>
export VERTEXAI_LOCATION=global
```

See [docs/llm.md](docs/llm.md) for more details on supported LLM providers,
thinking controls, and calling arbitrary LLM endpoint via out-of-process
bridge.

## Workflow

### Running the Web Server

Start the long-lived server. It owns the data directory (DB, observer),
recovers interrupted runs, and serves both the JSON API and the UI:

```bash
amplio serve
```

The server prints a URL with an access token on startup. The bind address is
`--listen` > `$AMPLIO_LISTEN` > `config.toml [listen]`. Starting, stopping, and
resuming runs can all be done from the web UI.

The default listen address is `0.0.0.0:26759` — reachable from any host
that can route to the machine, convenient for a corp network environment
(workstation + corp laptop). 

For solo single-machine use or SSH-tunneled access, bind to loopback instead
(`localhost:26759`) so amplio is only reachable locally or through the 
authenticated SSH tunnel. In this case, setting up TLS is recommended,
as modern browsers have 6-connection HTTP/1.1 cap on a single origin,
amplio UI may stop auto updating when multiple tabs are open, 
see [docs/tls.md](docs/tls.md).

Note: the printed URL contains a token for authentication — clicking on the
link directly will authenticate you and save the auth info in a cookie.
Without the token or cookie, a readonly view is shown. So in a corp
environment, you can share the link of your running server with a teammate,
but do **not** share the printed-out token.

### Autonomous Run vs Interactive Run

When starting a new Run, we can choose to start it in two modes:

* **Autonomous Mode**: an autonomous `main-agent` will be started
  to work on the given task, it will run until task completion (or crash / manual
  cancellation). When concluded, a run report will be generated. The user can
  read the run report, or start a companion chatbot to investigate the outcome.
  In the run report, the user can also provide a follow up prompt to start a new
  iteration of follow up (autonomous) run.
* **Interactive Mode**: the initial text is just the first message to the
  chatbot. The operator work interactively with the chatbot to drive the work.
  The chatbot is fully agent, can do the task by itself or spawn sub-agents to
  do focused long running tasks. A run report will not be auto-generated in this
  mode because there is no natural completion point, but the operator can manually
  trigger it via the generation report button on the Web UI.

## Multi-instances

One amplio server can host multiple runs. If for some reason you want to run multiple
independent instances, you can specify a different listening port and data dir via:

```bash
AMPLIO_DATA_DIR=</path/to/data/dir> AMPLIO_LISTEN=0.0.0.0:<PORT> amplio serve
```

## Headless Run (Eval Workflow)

Amplio support headless run, useful to be used as harness in eval or training. It
will start / resume a single run, and exit when the run completes.

Note: to run multiple headless runs on the same machine, use the multi-instance
setup above with different data dirs and ports.

```bash
amplio headless run --task="$(cat /path/to/task.md)" --workspace=/path/to/repo
```




To resume a crashed run (its LLM, workspace, and agent are reconstructed from
the run's stored config, so no extra flags are needed):

```bash
amplio headless resume <run-id>
```

### AGENTS.md

Operator instructions injected into the agent's context at every session
bootstrap. Two source files, both optional, both snapshotted ONCE at
run-start and inherited by every agent in the run (root + sub-agents):

- `<data-dir>/AGENTS.md` — rules that apply to every run on this machine.
- `<workspace>/AGENTS.md` — rules that apply to runs started from this
  workspace. Read from the workspace the run was launched against.

If both exist, they're concatenated with section headers labeling the source
path, and stored as a single `SystemEvent` (marker `agents_md`) at step 0.
Mid-run edits don't affect ongoing runs — the system prompt is immutable
after bootstrap. Restart the run to pick up edits.

## Security Caveats

Note there is *no* sandboxing in Amplio's bash tool calls. It is recommended
to run Amplio inside a sandboxed environment when security is a concern.

## Disclaimer

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

## Citation

If you find Amplio helpful, please cite the following BibTeX:

```bibtex
@misc{Zhang2026Amplio,
  author       = {Chiyuan Zhang and Da Huang and Chen Liang and Andrew Li and {Amplio Contributors}},
  title        = {{Amplio: A Lightweight Agent Harness for Robust and Long-Horizon Runs}},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/google-deepmind/amplio}
}
```
