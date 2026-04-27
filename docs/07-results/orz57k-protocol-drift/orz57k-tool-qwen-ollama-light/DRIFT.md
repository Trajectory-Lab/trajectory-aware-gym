# Drift Note

This published set is archived instead of being kept under the formal
`orz57k-tool` results.

- `config_snapshot.yaml` uses `val_size=50`, while the canonical Orz57K tool
  protocol requires `val_size=300`.
- `eval_size=500` and `tasks_per_minibatch=3` match the formal protocol, but
  the validation split drift alone is enough to make the run non-canonical.
- `gepa_budget.mode` is `light`, so this is also not a formal medium-budget
  result for the summary tables.

Published timestamp retained here:

- `20260423T163304Z`
