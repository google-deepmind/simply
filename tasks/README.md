# Simply research tasks

Each subdirectory is a *task suite*: a set of research tasks with fixed
objectives, fixed evaluation, and a validator that turns a finished run into a
score. A task is written for an autonomous research agent: it states what may be
changed, what is frozen, how to launch the scored run, and how the result is
judged.

| Suite | Tasks | What it measures |
|---|---|---|
| [`research_bench/`](research_bench/) | 11 | LLM pretraining under a FLOP cap, optimizer time-to-target, RL post-training (math + tool use), porting a published architecture into simply, and test-time decoding efficiency |

Run everything from the repository root (the suites are not part of the
installed `simply` wheel; they import it):

```bash
cd /path/to/simply
python -m pytest tasks/research_bench/tests -q
```
