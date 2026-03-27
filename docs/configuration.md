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
settings.sagemaker.endpoint_1_7b # "qwen3-1-7b-base"
settings.experiment.random_seed  # 42
settings.logging.level           # "INFO"
settings.cost_tracking.enabled   # True
settings.fitness.gamma           # 0.99
settings.fitness.lambda_         # 0.1
```

## LLM Provider Factory

```python
from trajectory_aware_gym.config.llm_provider import get_task_lm, get_reflection_lm

# Task models (routed by name to Ollama, Bedrock, or SageMaker)
task_lm = get_task_lm("qwen3:1.7b", "train")            # Ollama, temp=1.0
eval_lm = get_task_lm("llama:8b", "eval")               # Bedrock, temp=0.0
sage_lm = get_task_lm("qwen3-sagemaker:1.7b", "train")  # SageMaker, temp=1.0

# Reflection model (Claude Sonnet 4.5 via Bedrock)
reflection_lm = get_reflection_lm()

# With DSPy
import dspy
dspy.configure(lm=get_task_lm("qwen3:1.7b", "train"))
```

### Available Models

| Name | Provider | Model |
|------|----------|-------|
| `qwen3:1.7b` | Ollama (local) | Qwen3 1.7B Base |
| `qwen3:4b` | Ollama (local) | Qwen3 4B Base |
| `qwen3-sagemaker:1.7b` | SageMaker (AWS) | Qwen3 1.7B Base |
| `qwen3-sagemaker:4b` | SageMaker (AWS) | Qwen3 4B Base |
| `llama:1b` | Bedrock (AWS) | Llama 3.2 1B |
| `llama:3b` | Bedrock (AWS) | Llama 3.2 3B |
| `llama:8b` | Bedrock (AWS) | Llama 3.1 8B |

Ollama models require local setup. See [docs/ollama_setup.md](ollama_setup.md) for installation.
SageMaker models require a running endpoint. See [SageMaker Endpoints](#sagemaker-endpoints) below for deploy/teardown instructions.

### Model Roles

| Role | Purpose | Current Models |
|------|---------|----------------|
| **Task model** | Runs in GEM environments, optimized by GEPA | Qwen3 1.7B/4B Base (Ollama or SageMaker), Llama 1B/3B/8B (Bedrock) |
| **Reflection model** | GEPA prompt mutation and reflection | GPT OSS 120B (Bedrock) |

## Env Var Override Convention

Every YAML field can be overridden via an env var named `PREFIX_FIELD`:

| YAML Section | Env Prefix | Example |
|-------------|------------|---------|
| `aws` | `AWS_` | `AWS_REGION=eu-west-1` |
| `ollama` | `OLLAMA_` | `OLLAMA_API_BASE=http://host:11434` |
| `sagemaker` | `SAGEMAKER_` | `SAGEMAKER_REGION=us-west-2` |
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
| `bedrock_claude_sonnet_4_5` | `str` | Bedrock reflection model ID (legacy field name retained for compatibility) |
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

### `sagemaker` — SageMaker Custom Endpoints

| Field | Type | Description |
|-------|------|-------------|
| `region` | `str` | AWS region for SageMaker endpoints |
| `role_arn` | `str` | IAM execution role ARN for SageMaker |
| `instance_type` | `str` | EC2 instance type (e.g. `ml.g5.xlarge`) |
| `tgi_image_uri` | `str` | HuggingFace TGI container image URI |
| `endpoint_1_7b` | `str` | SageMaker endpoint name for Qwen3 1.7B Base |
| `endpoint_4b` | `str` | SageMaker endpoint name for Qwen3 4B Base |
| `model_id_1_7b` | `str` | HuggingFace model ID for Qwen3 1.7B Base |
| `model_id_4b` | `str` | HuggingFace model ID for Qwen3 4B Base |

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

### Dry-Run Note

The current `math-dry-run` experiment closes the K4 integration milestone by validating the full GEPA + DSPy + GEM execution path with bounded budget and saved artifacts. It is still a smoke-test configuration:
- small train/validation samples
- bounded GEPA metric calls
- short episode cap
- no guarantee of multi-step or tool-using behavior in every run

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

## SageMaker Endpoints

Qwen3 Base models are not natively hosted on Bedrock, so we deploy them to
SageMaker using HuggingFace TGI containers. Once deployed, they are callable
through the same `get_task_lm()` interface as all other models.

### Prerequisites

Each team member needs their own AWS credentials configured:

```bash
aws configure
# or set in .env:
#   AWS_ACCESS_KEY_ID=...
#   AWS_SECRET_ACCESS_KEY=...
```

The IAM user must have permissions for `sagemaker:*` and `iam:PassRole` on the
execution role.

### Deploy / Teardown

```bash
# Deploy endpoints (~5-10 min startup, ~$1.41/hr per endpoint)
uv run python -m trajectory_aware_gym.sagemaker.deploy deploy 1.7b
uv run python -m trajectory_aware_gym.sagemaker.deploy deploy 4b
uv run python -m trajectory_aware_gym.sagemaker.deploy deploy all   # both

# Check status and cost
uv run python -m trajectory_aware_gym.sagemaker.deploy status

# Shut down when done (stops billing)
uv run python -m trajectory_aware_gym.sagemaker.deploy delete all
uv run python -m trajectory_aware_gym.sagemaker.deploy delete 1.7b  # one only
```

### Quick Test (standalone client)

```bash
uv run python -m trajectory_aware_gym.sagemaker.client 1.7b "Hello!"
uv run python -m trajectory_aware_gym.sagemaker.client 4b "Hello!"
```

### Usage via Unified Interface

SageMaker models use the same `get_task_lm()` factory as Ollama and Bedrock
models. Just pass a different model name:

```python
from trajectory_aware_gym.config.llm_provider import get_task_lm

# These all return dspy.LM — same interface, different backends
ollama_lm    = get_task_lm("qwen3:1.7b", "train")            # Ollama
bedrock_lm   = get_task_lm("llama:8b", "eval")               # Bedrock
sagemaker_lm = get_task_lm("qwen3-sagemaker:1.7b", "train")  # SageMaker

# Use with DSPy exactly as before
import dspy
dspy.configure(lm=sagemaker_lm)
```

### Cost

| Instance | GPU | Cost |
|----------|-----|------|
| `ml.g5.xlarge` | 1x A10G (24 GB) | ~$1.41/hr |

Running both endpoints: ~$2.82/hr (~$67.68/day). **Always shut down when done.**
