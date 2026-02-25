# Integration Test Matrix (Phase 2)

This matrix defines the minimum automated integration coverage for GEM environment episodes, tool/config routing, and optimizer-adjacent execution pathways.

## Scope

- Environment + trajectory capture lifecycle
- Tool/provider routing for task and reflection models
- Configuration coupling between model provider and runtime settings
- CI-runnable tests that avoid network/API dependencies

## Matrix

| ID | Area | Scenario | Automated | Test Location | Notes |
|---|---|---|---|---|---|
| IT-01 | Env + Adapter | Episode trajectory is captured, validated, and persisted to `logs/` | Yes | `tests/integration/test_system_pipeline.py::test_episode_trajectory_persists_to_logs` | Uses `TrajectoryLogger` with temp project paths |
| IT-02 | Env + Adapter | Persisted trajectory can be reconstructed from JSON schema | Yes | `tests/integration/test_system_pipeline.py::test_saved_trajectory_round_trip_validation` | Validates schema round-trip with `TrajectoryLog` |
| IT-03 | Tools + Config | Task model routing selects provider model and train/eval temperature for all 5 variants | Yes | `tests/integration/test_system_pipeline.py::test_task_model_routing_and_temperature` | Parametrized across Ollama (qwen3:1.7b, 4b) and Bedrock (llama:1b, 3b, 8b) |
| IT-04 | Tools + Config | Unrecognised model name returns `None` without calling provider | Yes | `tests/integration/test_system_pipeline.py::test_task_model_unrecognised_name_returns_none` | Documents match/case fall-through behaviour |
| IT-05 | Tools + Config | Reflection model uses configured GEPA reflection model on Bedrock path | Yes | `tests/integration/test_system_pipeline.py::test_reflection_model_routing` | Verifies reflection LM kwargs |

## Supporting Unit Tests

These unit tests cover branch paths exercised by the integration scenarios above:

| Area | Test Location | Coverage |
|---|---|---|
| AWS client config credential inclusion/exclusion | `tests/unit/test_aws_config_clients.py::test_bedrock_client_config` | Parametrized: empty, full, and partial credential scenarios |
| S3 config delegation to Bedrock config | `tests/unit/test_aws_config_clients.py::test_s3_client_config_delegates_to_bedrock_payload` | Verifies reuse of credential logic |

## Execution

Run all tests:

```bash
uv run poe test
```

Run integration tests only:

```bash
uv run pytest tests/integration -v
```
