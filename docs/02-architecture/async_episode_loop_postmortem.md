# Async Episode Loop — Post-Mortem (#127)

**Outcome:** Reverted. The async refactor was implemented, tested, and then undone
after discovering it added complexity without benefiting the actual execution path.

## Original Goal

Eliminate the sync-over-async `_run_sync()` workaround in `ToolRuntime` by making the
episode loop async-native (`acompletion`, `async def run_episode`, etc.), then use
`asyncio.gather` for concurrent GEPA episode evaluation (#140).

## What Was Built

- `ToolRuntime`: async `execute()` / `list_schemas()` with `asyncio.wait_for` timeouts
- `GEMEpisodeRunner`: all methods async, `litellm.acompletion` instead of `completion`
- `GEMSolverModule`: `aforward()` + sync `forward()` wrapping `asyncio.run(aforward())`
- CLI script: async `run_episode()` / `run_smoke_episode()`
- `pytest-asyncio` configured with `asyncio_mode = "auto"`
- Hand-rolled `GEPAOptimizer` with `asyncio.gather`-based `aoptimize()`

## Why It Was Reverted

### DSPy GEPA owns parallelism

`dspy.GEPA.compile()` → `DspyAdapter.evaluate()` → `dspy.Evaluate` → `ParallelExecutor`
which uses `ThreadPoolExecutor(num_threads)`. Each thread calls `program(**example.inputs())`
→ `forward()`. DSPy has no async evaluation path — `Evaluate` never calls `acall()`.

This means our `forward()` → `asyncio.run(aforward())` created and destroyed an event
loop per thread per episode. Functionally identical to just calling sync `completion()`
directly — no connection pooling, no concurrent I/O benefit.

### The async refactor introduced the LiteLLM warning

Repeated `asyncio.run(litellm.acompletion(...))` triggered:
```
RuntimeWarning: coroutine 'Logging.async_success_handler' was never awaited
```
Sync `litellm.completion()` doesn't have this issue. The async refactor *created* the
warning it was trying to work around.

### The hand-rolled optimizer was dead code

The `asyncio.gather` concurrency was built into a custom `GEPAOptimizer` that was never
used in the production path — only in unit tests. The real optimizer (`dspy.GEPA`) was
always the one running experiments.

### Background event loop was unsafe

An attempt to bridge async into DSPy's threaded model via a `BackgroundAsyncRunner`
(long-lived event loop on a daemon thread) failed because DSPy GEPA uses forked
parallelism internally, which is unsafe with shared thread state.

## Current Architecture

Everything is sync. `ToolRuntime._run_sync()` handles the one unavoidable async
boundary (FastMCP's `call_tool()` is async) with a thread fallback.

```
dspy.GEPA.compile()
  → ThreadPoolExecutor(num_threads)
    → GEMSolverModule.forward()
      → GEMEpisodeRunner.run()          [sync]
        → litellm.completion()           [sync]
        → ToolRuntime.execute()          [sync, _run_sync for MCP]
```

## Lessons

1. **Understand the caller before optimizing the callee.** The async refactor assumed
   we'd control the event loop. DSPy controls it via thread pool.
2. **`asyncio.run()` per call ≠ async benefit.** Without a long-lived loop, `acompletion`
   is just `completion` with extra overhead.
3. **Don't build infrastructure for a caller that doesn't exist.** The hand-rolled
   optimizer existed to justify the async work, not the other way around.

## When to Revisit

- If DSPy adds `async def evaluate()` or `acall()`-based GEPA evaluation
- If we drop DSPy GEPA and write our own optimization loop with a single `asyncio.run()` at the top
- If episode execution moves to a server context (FastAPI, etc.) with a persistent event loop
