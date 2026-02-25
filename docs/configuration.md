# Configuration

All configuration lives in two places:

| Source | Purpose |
|--------|---------|
| `config/trajectory-aware-gym.yaml` | Non-sensitive defaults (checked into git) |
| `.env` | Secrets and per-developer overrides |

**Priority:** `.env` / env vars override YAML values.

## How It Works

`Settings` is a plain Python class that loads once (singleton). On first access:

1. `.env` is loaded into `os.environ` via `python-dotenv`
2. `config/trajectory-aware-gym.yaml` is parsed via PyYAML
3. For each config section, env vars override YAML values using `PREFIX_FIELD` naming

```
.env (secrets)  ──→  os.environ  ──→  Settings (singleton)
                                          ↑
trajectory-aware-gym.yaml (defaults) ─────┘
```

## Accessing Configuration

```python
from trajectory_aware_gym.config import settings

# Sub-config sections
settings.aws.region              # "us-east-1"
settings.gem.max_steps           # 50
settings.gepa.budget             # "medium"
settings.ollama.api_base         # "http://localhost:11434"
settings.experiment.random_seed  # 42
settings.logging.level           # "INFO"
settings.cost_tracking.enabled   # True
settings.fitness.gamma           # 0.99
settings.fitness.lambda_         # 0.1
```

## LLM Provider Factory

```python
from trajectory_aware_gym.config.llm_provider import get_task_lm, get_reflection_lm

# Task models (routed by name to Ollama or Bedrock)
task_lm = get_task_lm("qwen3:1.7b", "train")  # Ollama, temp=1.0
eval_lm = get_task_lm("llama:8b", "eval")      # Bedrock, temp=0.0

# Reflection model (Claude Sonnet 4.5 via Bedrock)
reflection_lm = get_reflection_lm()

# With DSPy
import dspy
dspy.configure(lm=get_task_lm("qwen3:1.7b", "train"))
```

### Available Models

| Name | Provider | Model |
|------|----------|-------|
| `qwen3:1.7b` | Ollama (local) | Qwen3 1.7B |
| `qwen3:4b` | Ollama (local) | Qwen3 4B |
| `llama:1b` | Bedrock (AWS) | Llama 3.2 1B |
| `llama:3b` | Bedrock (AWS) | Llama 3.2 3B |
| `llama:8b` | Bedrock (AWS) | Llama 3.1 8B |

### Model Roles

| Role | Purpose | Current Models |
|------|---------|----------------|
| **Task model** | Runs in GEM environments, optimized by GEPA | Qwen3 1.7B/4B (Ollama), Llama 1B/3B/8B (Bedrock) |
| **Reflection model** | GEPA prompt mutation and reflection | Claude Sonnet 4.5 (Bedrock) |

## Env Var Override Convention

Every YAML field can be overridden via an env var named `PREFIX_FIELD`:

| YAML Section | Env Prefix | Example |
|-------------|------------|---------|
| `aws` | `AWS_` | `AWS_REGION=eu-west-1` |
| `ollama` | `OLLAMA_` | `OLLAMA_API_BASE=http://host:11434` |
| `gem` | `GEM_` | `GEM_MAX_STEPS=100` |
| `gepa` | `GEPA_` | `GEPA_BUDGET=heavy` |
| `experiment` | `EXPERIMENT_` | `EXPERIMENT_RANDOM_SEED=123` |
| `logging` | `LOG_` | `LOG_LEVEL=DEBUG` |
| `cost_tracking` | `COST_TRACKING_` | `COST_TRACKING_ENABLED=false` |
| `fitness` | `FITNESS_` | `FITNESS_GAMMA=0.95` |

Type coercion is automatic: `"42"` → `int`, `"3.14"` → `float`, `"true"` → `bool`.

## Config Sections

### `aws` — AWS and Bedrock

| Field | Type | Description |
|-------|------|-------------|
| `region` | `str` | AWS region |
| `access_key_id` | `str` | AWS access key (secret, from `.env`) |
| `secret_access_key` | `str` | AWS secret key (secret, from `.env`) |
| `session_token` | `str` | Optional session token (secret, from `.env`) |
| `bedrock_claude_sonnet_4_5` | `str` | Bedrock model ID for Claude |
| `bedrock_llama_1b` | `str` | Bedrock model ID for Llama 1B |
| `bedrock_llama_3b` | `str` | Bedrock model ID for Llama 3B |
| `bedrock_llama_8b` | `str` | Bedrock model ID for Llama 8B |
| `s3_bucket` | `str` | S3 bucket for results |
| `s3_prefix` | `str` | S3 key prefix |

### `ollama` — Local Ollama

| Field | Type | Description |
|-------|------|-------------|
| `api_base` | `str` | Ollama API URL |
| `task_model_1_7b` | `str` | LiteLLM model string for Qwen3 1.7B |
| `task_model_4b` | `str` | LiteLLM model string for Qwen3 4B |

### `gem` — GEM Environment

| Field | Type | Description |
|-------|------|-------------|
| `max_steps` | `int` | Max steps per episode |
| `temperature_train` | `float` | Temperature during training (1.0) |
| `temperature_eval` | `float` | Temperature during evaluation (0.0) |

### `gepa` — GEPA Optimizer

| Field | Type | Description |
|-------|------|-------------|
| `budget` | `Literal["light", "medium", "heavy"]` | Compute budget |
| `population_size` | `int` | Number of prompts in population |
| `iterations` | `int` | Optimization iterations |
| `reflection_model` | `str` | Bedrock model ID for reflection |

### `experiment` — Experiment

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Experiment name |
| `random_seed` | `int` | Random seed |
| `num_replications` | `int` | Number of replications |

### `logging` — Logging

| Field | Type | Description |
|-------|------|-------------|
| `level` | `str` | Log level |
| `file` | `str` | Log file path |

### `cost_tracking` — Cost Tracking

| Field | Type | Description |
|-------|------|-------------|
| `enabled` | `bool` | Enable cost tracking |
| `alert_threshold` | `float` | Cost alert threshold ($) |

### `fitness` — Trajectory-Aware Fitness

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `gamma` | `float` | [0.0, 1.0] | Reverse-time discount factor |
| `lambda` | `float` | >= 0.0 | Auxiliary per-turn reward scaling |
| `loop_penalty_weight` | `float` | >= 0.0 | Loop detection penalty weight |
| `step_efficiency_weight` | `float` | >= 0.0 | Step efficiency bonus weight |
| `max_steps` | `int` | >= 1 | Max steps for efficiency normalization |
| `loop_window` | `int` | >= 1 | Sliding window for loop detection |

Note: The YAML field is `lambda` but the Python attribute is `lambda_` (Python keyword).
Env var override: `FITNESS_LAMBDA`.

## Using Config in Production and Experiment Code

Sub-config models are Pydantic `BaseModel` instances. Use `model_copy(update={...})`
to derive variants from the YAML defaults without mutating the singleton:

```python
from trajectory_aware_gym.config import settings
from trajectory_aware_gym.fitness import CompositeFitness

# Production: use YAML defaults directly
fitness = CompositeFitness()  # internally uses settings.fitness

# Explicit: pass config from singleton
fitness = CompositeFitness(settings.fitness)

# Experiment override: tweak specific hyperparameters
custom = settings.fitness.model_copy(update={"gamma": 0.95, "lambda_": 0.2})
fitness = CompositeFitness(custom)

# Ablation: disable a term by zeroing its weight
no_loop = settings.fitness.model_copy(update={"loop_penalty_weight": 0.0})
fitness = CompositeFitness(no_loop)

# Ablation: only discounted return (disable all auxiliary terms)
dr_only = settings.fitness.model_copy(update={
    "loop_penalty_weight": 0.0,
    "step_efficiency_weight": 0.0,
})
fitness = CompositeFitness(dr_only)
```

This pattern works for any sub-config model (e.g., `settings.gem.model_copy(update={...})`).

## Adding a New Model

1. **YAML**: Add the model ID to the appropriate section in `config/trajectory-aware-gym.yaml`
2. **Model class**: Add the field to the corresponding model in `config/core.py`
3. **LLM provider**: Add the model name to `TaskModelName` and a `case` branch in `get_task_lm()` in `config/llm_provider.py`
4. **Verify**: Run `poe test` to confirm config loads correctly

## Adding a New Config Section

1. **YAML**: Add the section to `config/trajectory-aware-gym.yaml`
2. **Model**: Create a new `BaseModel` subclass in `config/core.py`
3. **Section map**: Add a `(yaml_key, env_prefix, ModelClass)` tuple to `_SECTION_MAP`
4. **Settings**: Add the typed class attribute and `@property` to `Settings`
5. **Exports**: Add the model to `config/__init__.py`

## AWS Credential Validation

AWS credentials are not validated at load time (since not all code paths need Bedrock). Call `settings.validate_aws()` before making Bedrock API calls:

```python
from trajectory_aware_gym.config import settings

settings.validate_aws()  # Raises ValueError if creds missing but Bedrock configured
```
