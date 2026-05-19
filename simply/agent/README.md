## Quick Start

Simply Agent is a minimal agent harness with Bash tool and
context management, for long-running research tasks.

### 1. Install

```bash
pip install ".[agent]"
```

### 2. Configure LLM access

The agent uses [LiteLLM](https://github.com/BerriAI/litellm)
as the LLM gateway. Any LLM provider supported by LiteLLM
can be used. Set the appropriate environment variable for
your provider:

```bash
# Option A: Google Cloud Vertex AI
gcloud auth application-default login
export VERTEXAI_PROJECT=your-project
export VERTEXAI_LOCATION=global

# Option B: Gemini API (https://aistudio.google.com/apikey)
export GEMINI_API_KEY=your-key

# Option C: OpenAI (https://platform.openai.com/api-keys)
export OPENAI_API_KEY=your-key
```

### 3. Run the example

```bash
# Using Vertex AI:
python -m simply.agent.main \
    --task_file=simply/agent/example_tasks/code_stats.md \
    --env="Local:." \
    --llm="LiteLLM:vertex_ai/gemini-2.5-pro"

# Using Gemini API:
python -m simply.agent.main \
    --task_file=simply/agent/example_tasks/code_stats.md \
    --env="Local:." \
    --llm="LiteLLM:gemini/gemini-2.5-pro"

# Using Claude on Vertex AI:
python -m simply.agent.main \
    --task_file=simply/agent/example_tasks/code_stats.md \
    --env="Local:." \
    --llm="LiteLLM:vertex_ai/claude-sonnet-4-6"

# Using OpenAI:
python -m simply.agent.main \
    --task_file=simply/agent/example_tasks/code_stats.md \
    --env="Local:." \
    --llm="LiteLLM:openai/gpt-4o"
```

This runs a small task that scans Python files in the current
directory and reports class usage statistics. It completes in
under a minute.

### 4. View the trajectory

The agent writes detailed HTML trajectories to the session
directory (`~/.simply_agent/sessions/` by default):

```bash
python3 -m http.server 9999 -d ~/.simply_agent/sessions/
```

## Reference

See `main.py` for all command-line flags, including
`--base_system_dir`, `--resume`, and `--display_full`.
