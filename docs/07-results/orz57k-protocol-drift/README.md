# Orz57K Protocol Drift Archive

This directory keeps published Orz57K/Qwen/Ollama artifacts that should not sit
beside the formal `orz57k-tool` results.

- `summary.md` ignores everything under this archive.
- These artifacts are retained for provenance, not for the formal Orz57K result
  tables.

Archived groups:

- `orz57k-tool-qwen-ollama`: core `300/500/3` settings match the Orz57K tool
  protocol, but the publish came from an ad hoc config name and a `light`
  budget.
- `orz57k-tool-qwen-ollama-light`: `val_size=50` instead of the canonical
  `300`, and the run also used a `light` budget.
- `orz57k-tool-qwen-ollama-medium`: `val_size=50` instead of `300`, and the
  published run includes replication failures/timeouts.
