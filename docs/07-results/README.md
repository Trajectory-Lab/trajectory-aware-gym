# Results

Experiment results, analysis outputs, and statistical comparisons.

## HotpotQA - Tool-augmented prompting
============================================================
  Replication complete: Llama-3.1-8B-Instruct seed=42
  Validation: Baseline accuracy:  7.3% (22/300) (HotpotQA)
  Validation: Optimized accuracy: 31.7% (95/300) (HotpotQA)
  Test/eval: Baseline accuracy:  38.5% (5/13) (hotpotqa)
  Test/eval: Optimized accuracy: 41.8% (164/392) (hotpotqa)
  Total cost:         $0.9696
  Total tokens:       3,737,457
  Elapsed:            4625.5s
============================================================

```
uv run poe run-experiment  --config experiments/hotpotqa-tool/config.yaml --models Llama-3.1-8B-Instruct --seeds 42 --max-metric-calls 1500 --fresh
```

============================================================
  Replication complete: Llama-3.1-8B-Instruct seed=42
  Validation: Baseline accuracy:  30.3% (91/300) (HotpotQA)
  Validation: Optimized accuracy: 30.3% (91/300) (HotpotQA)
  Test/eval: Baseline accuracy:  39.5% (502/1270) (hotpotqa)
  Test/eval: Optimized accuracy: 41.6% (528/1269) (hotpotqa)
  Total cost:         $0.3576
  Total tokens:       1,369,121
  Elapsed:            1701.7s
============================================================

```
uv run poe run-experiment  --config experiments/hotpotqa-notool/config.yaml --models Llama-3.1-8B-Instruct --seeds 42 --max-metric-calls 1500 --fresh
```
