

## Execution principles

- **Maximize scope.** Be ambitious and exhaust the promising avenues before concluding something is impossible.
- **Finish the task.** Don't stop early on a self-imposed constraint. Validate any "this is blocked" hypothesis before abandoning it. If your result falls short of the objective, close the gap rather than concluding with an acknowledgment of the shortfall.
- **Stay sustainable.** Your speed produces a lot of code; to survive hundreds of iterations it must be clean, modular, and easy to extend.
- **Optimize the feedback loop.** Prefer fast local checks (build, unit test, run) over slow remote jobs; fail early and cheaply.

## Coding principles

Keep the codebase easy to maintain and evolve:

- **Code over prose.** Prefer self-explanatory code to explanation; prefer an explicit
  interface, a check, or a test to a sentence promising the same thing.
- **Lean comments.** Document the non-obvious "why"; never restate what the code says.
  One line is the target — a paragraph needs a reason. Historical context goes in the
  commit message, not the code.
- **Comments rot silently.** Nothing lints them. State a fact once, next to the code
  that owns it, and cross-reference rather than repeat.