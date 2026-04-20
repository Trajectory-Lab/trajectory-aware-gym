# Common Issues

Known issues, workarounds, and gotchas encountered during development.

---

## Template

```markdown
### [Short title]

**Symptom:** What you see (error message, hang, wrong output).

**Cause:** Why it happens.

**Fix / Workaround:** What to do about it.

**Affected components:** Which files, tools, or environments are involved.
```

---

### Qwen3 thinking mode produces empty `content` with low `max_tokens`

**Symptom:** LiteLLM `completion()` returns an empty string in
`response.choices[0].message.content` even though `usage.total_tokens` shows
the model generated tokens. The action logged is `[empty-action]`.

**Cause:** Qwen3 models use `<think>...</think>` reasoning by default. The
thinking tokens consume the `max_tokens` budget first. If the budget is too
small (e.g. 512), the model exhausts it entirely on reasoning and never
produces a visible answer. The reasoning text lands in
`response.choices[0].message.reasoning_content`, not `.content`.

**Fix / Workaround:**
1. Set `max_tokens` high enough for the model to finish thinking *and* answer
   (2048+ for math problems).
2. When `.content` is empty, fall back to `.reasoning_content` so the raw
   thinking chain is still submitted to the environment as the action.

**Affected components:** `scripts/run_gem_episode.py` (`generate_smoke_action`),
any future code calling Qwen3 via LiteLLM's `ollama/` provider.

---


### GEM MathEnv `close()` fails with `use_mp=False`

**Symptom:** `AttributeError: 'MathEnv' object has no attribute 'mp_pool'`
when calling `env.close()`.

**Cause:** `MathEnv.__init__` only creates `self.mp_pool` when `use_mp=True`
(the default). The `close()` method unconditionally accesses `self.mp_pool`
without checking.

**Fix / Workaround:** Either keep `use_mp=True` (the default), or wrap
`env.close()` in a try/except. This is a bug in the upstream `gem` package.

**Affected components:** `gem.envs.math_env.MathEnv.close()`.

---

### GEM math_grader `SyntaxWarning` on Python 3.13+

**Symptom:** Multiple `SyntaxWarning: invalid escape sequence` warnings from
`gem/utils/math_grader.py` on startup.

**Cause:** The upstream GEM package uses unescaped backslashes in regex strings
(e.g. `"\{"` instead of `"\\{"`). Python 3.13 warns on these; a future version
will make them errors.

**Fix / Workaround:** Ignore — these are cosmetic warnings from a vendored
dependency. They do not affect correctness.

**Affected components:** `gem.utils.math_grader` (upstream).

---

### `fork()` deprecation warning from GEM's multiprocessing pool

**Symptom:** `DeprecationWarning: This process (pid=...) is multi-threaded,
use of fork() may lead to deadlocks in the child.`

**Cause:** `MathEnv` creates a `multiprocessing.Pool(1)` for answer-grading
timeouts. Python 3.13 warns when forking in a multi-threaded process.

**Fix / Workaround:** Ignore — this is a cosmetic warning. The pool is only
used for a 1-second grading timeout and works correctly in practice.

**Affected components:** `gem.envs.math_env.MathEnv.__init__` (upstream).

---

### GEM `QaEnv.step()` raises `UnboundLocalError` when no answer is extractable

**Symptom:** `UnboundLocalError: cannot access local variable 'is_correct'
where it is not associated with a value` originating from
`gem/envs/qa_env.py:100`. Seen during HotpotQA / QA evaluation when the model
output doesn't contain the required `<answer>...</answer>` tag (or
`\boxed{...}` when `extract_boxed=True`).

**Cause:** Upstream `QaEnv.step` only binds `is_correct` inside the `else`
branch that runs when the extractor finds an answer. If the extractor returns
`None`, the method sets `reward = 0.0` but leaves `is_correct` unbound, then
reads it unconditionally in the final `return`.

**Fix / Workaround:** Patched at runtime by
`trajectory_aware_gym.adapters.gem_env_factory._apply_upstream_patches`, which
replaces `QaEnv.step` with a version that defaults `is_correct = False`. The
patch is applied the first time `make_env` runs and is idempotent.

**Affected components:** `gem.envs.qa_env.QaEnv.step` (upstream).
