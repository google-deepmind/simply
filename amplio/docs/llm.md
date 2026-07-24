# LLM Providers

Amplio resolves a model from a **spec string** of the form:

```
<provider>:<model>[?k=v&k=v…]
```

`createProvider` (cmd/amplio) splits on the first `:` to pick the provider
"class" from the registry, then splits the remainder on `?` into the model id and
an optional query of per-model **args** that are forwarded to the request. Each
provider family interprets the args (see [Thinking](#thinking-control) below).

## Providers

| Prefix          | Family / backend                                 | Auth                                  |
| --------------- | ------------------------------------------------ | ------------------------------------- |
| `vertex-claude` | Claude on Vertex AI (`anthropic-sdk-go`)         | ADC + `VERTEXAI_PROJECT`/`VERTEXAI_LOCATION` |
| `claude`        | Claude direct Anthropic API (`anthropic-sdk-go`) | `ANTHROPIC_API_KEY`; no GCP |
| `vertex-gemini` | Gemini on Vertex AI (`google.golang.org/genai`)  | ADC + `VERTEXAI_PROJECT`/`VERTEXAI_LOCATION` |
| `gemini`        | Gemini Developer API (`genai`)                   | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`); no GCP |
| `subprocess`    | out-of-process bridge over a unix socket         | whatever the bridge needs (e.g. Beyond/ADC)    |

All providers default to `MaxTokens` (output cap) **65536**. The Claude provider
always streams under the hood (even for blocking `Call`s) because Anthropic
rejects non-streaming requests whose `max_tokens` could exceed the 10-minute
limit.

Specs are configured in `config.toml` (`[run] llms`, `system_llm_hq`,
`system_llm_fast`) — see [config.toml](#example-configtoml).

## Thinking control

There is **no unified thinking interface** — thinking knobs differ by model
generation, so the spec args are passed through per family. The model itself
validates them, so a wrong knob returns a clear API error (e.g. opus-4-8 rejects
`thinking.type=enabled` with *"…not supported for this model. Use
thinking.type.adaptive and output_config.effort…"*).

### Claude (`vertex-claude:`)

Args are injected **verbatim** into the request body as raw JSON, using dotted
paths (so `thinking.budget_tokens` nests). Values are coerced to int → bool →
float → string. Use whatever the Anthropic API supports for that model; no
amplio code change is needed when Anthropic adds a knob.

- **Newer models — opus-4-7, opus-4-8** use *adaptive* thinking plus an effort
  level (explicit `budget_tokens` is **rejected**):

  ```
  vertex-claude:claude-opus-4-8?thinking.type=adaptive&output_config.effort=high
  ```

  `output_config.effort` ∈ `low | medium | high | xhigh | max`.

- **Older models — opus-4-6, sonnet-4-6** use *enabled* thinking with an explicit
  token budget (they also accept adaptive):

  ```
  vertex-claude:claude-opus-4-6?thinking.type=enabled&thinking.budget_tokens=2048
  ```

  `budget_tokens` must be ≥ 1024 and < `max_tokens`.

### Gemini (`vertex-gemini:` / `gemini:`)

`genai` is a typed SDK (no raw passthrough), so a **known key set** is mapped;
an unknown key fails fast at construction:

| arg                | maps to                          |
| ------------------ | -------------------------------- |
| `thinking_budget`  | `ThinkingConfig.ThinkingBudget` (int; `0` disables where supported, `-1` dynamic) |
| `include_thoughts` | `ThinkingConfig.IncludeThoughts` (bool; default `true`) |
| `temperature`      | `GenerateContentConfig.Temperature` (float) |

```
vertex-gemini:gemini-3.5-flash?thinking_budget=2048&include_thoughts=true
```

## Verified configurations

Tested live (Vertex), single-turn **and** multi-turn with tool calls:

| spec                                                                          | thinking |
| ----------------------------------------------------------------------------- | -------- |
| `vertex-claude:claude-opus-4-8?thinking.type=adaptive&output_config.effort=high` | adaptive + effort |
| `vertex-claude:claude-opus-4-7?thinking.type=adaptive&output_config.effort=high` | adaptive + effort |
| `vertex-claude:claude-opus-4-6?thinking.type=enabled&thinking.budget_tokens=2048` | enabled + budget |
| `vertex-claude:claude-sonnet-4-6?thinking.type=enabled&thinking.budget_tokens=2048` | enabled + budget |
| `vertex-gemini:gemini-3.5-flash?thinking_budget=2048&include_thoughts=true`    | budget |
| `vertex-gemini:gemini-3.1-pro-preview?thinking_budget=2048&include_thoughts=true` | budget |

## Subprocess bridges (`subprocess:`)

For backends whose SDKs shouldn't be linked into the main binary (notably
corp-only APIs like **Beyond**), `subprocess:/path/to/bridge?model=NAME[&k=v]`
spawns an out-of-process bridge and talks to it over a tiny HTTP/NDJSON protocol
on a Unix domain socket. amplio manages the lifecycle: one long-lived process
per spec (reused across runs), crash-restart with a one-shot retry, health-poll
readiness, graceful reap on exit (+ `Pdeathsig` on Linux).

The bridge can be written in any language; `bridges/bridge.py` is a stdlib-only
reference with an `echo` backend (used for smoke testing) — see
[bridges/README.md](../bridges/README.md) for the protocol and how to add
your own backend. Example spec for the bundled echo backend:

```
subprocess:/path/to/bridge.py?model=echo-model&backend=echo
```



## Example config.toml

```toml
system_llm_hq   = "vertex-claude:claude-opus-4-8?thinking.type=adaptive&output_config.effort=high"
system_llm_fast = "vertex-gemini:gemini-3.5-flash?thinking_budget=0"   # 0 = no thinking (fast)

[run]
llms = [
  "vertex-claude:claude-opus-4-8?thinking.type=adaptive&output_config.effort=high",
  "vertex-claude:claude-opus-4-6?thinking.type=enabled&thinking.budget_tokens=4096",
  "vertex-gemini:gemini-3.1-pro-preview?thinking_budget=4096",
]
```

## Embedding providers

Recall (skill + lesson similarity search) needs a text embedder, configured
separately via `embed_model` (or `--embed-model` / `$AMPLIO_EMBED_MODEL`). 
The spec mirrors the LLM convention: `<backend>:<model>`.

`createEmbedder` (cmd/amplio) splits on the first `:` to pick the backend. A
bare model name with no `:` defaults to the `vertex` backend (back-compat with
older `embed_model` values).

| Backend  | Family / backend                            | Auth                                           |
| -------- | ------------------------------------------- | ---------------------------------------------- |
| `vertex` | Vertex AI embeddings (`google.golang.org/genai`) | ADC + `VERTEXAI_PROJECT`/`VERTEXAI_LOCATION` |
| `gemini` | Gemini Developer API embeddings (`genai`)   | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`); no GCP |

**Model availability differs by backend.** For example `text-embedding-005` is
Vertex-only, while `gemini-embedding-001` is available on both. Anthropic has no
embedding models, so there is no `claude` embedder backend.

The resolved `<backend>_<model>` is the **cache key** for stored vectors
(skills + lessons). If `embed_model` is unset, recall is disabled (no embedder is built).

```toml
embed_model = "vertex:text-embedding-005"   # or: gemini:gemini-embedding-001
```
