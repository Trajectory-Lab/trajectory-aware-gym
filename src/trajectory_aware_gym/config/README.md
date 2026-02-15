# Configuration Guide

## Usage

### Task models

```python
from trajectory_aware_gym.config.llm_provider import get_task_lm

# Qwen3 1.7B for training (local Ollama, temp=1.0)
task_lm = get_task_lm("qwen3:1.7b", "train")

# Qwen3 4B for evaluation (local Ollama, temp=0.0)
eval_lm = get_task_lm("qwen3:4b", "eval")

# Llama 8B for training (AWS Bedrock, temp=1.0)
llama_lm = get_task_lm("llama:8b", "train")
```

### Reflection model

```python
from trajectory_aware_gym.config.llm_provider import get_reflection_lm

# Claude Sonnet 4.5 (AWS Bedrock)
reflection_lm = get_reflection_lm()
```

### With DSPy

```python
import dspy
from trajectory_aware_gym.config.llm_provider import get_task_lm, get_reflection_lm

# Configure DSPy with task model
dspy.configure(lm=get_task_lm("qwen3:1.7b", "train"))

# GEPA with separate reflection model
optimizer = dspy.GEPA(
    metric=metric,
    reflection_lm=get_reflection_lm(),
)
```

### Available models

| Name | Provider | Model |
|------|----------|-------|
| `qwen3:1.7b` | Ollama (local) | Qwen3 1.7B |
| `qwen3:4b` | Ollama (local) | Qwen3 4B |
| `llama:1b` | Bedrock (AWS) | Llama 3.2 1B |
| `llama:3b` | Bedrock (AWS) | Llama 3.2 3B |
| `llama:8b` | Bedrock (AWS) | Llama 3.1 8B |

## Architecture

```
.env                  ← environment variables (secrets + model IDs)
aws_config.py         ← AWSConfig: Bedrock credentials, model IDs, S3
ollama_config.py      ← OllamaConfig: local Ollama API base + model IDs
settings.py           ← GEPAConfig, GEMConfig, ExperimentConfig, etc.
llm_provider.py       ← Factory: get_task_lm(), get_reflection_lm()
```

## Model Roles

| Role | Purpose | Current Models |
|------|---------|----------------|
| **Task model** | Runs in GEM environments, optimized by GEPA | Qwen3 1.7B/4B (Ollama), Llama 1B/3B/8B (Bedrock) |
| **Reflection model** | GEPA prompt mutation and reflection | Claude Sonnet 4.5 (Bedrock) |

## Adding a New Model

When adding a new LLM model to the project, update these files in order:

### 1. `.env` and `.env.example`

Add the model's environment variable:

```bash
# For Bedrock models:
BEDROCK_<MODEL_NAME>=<bedrock-model-id>

# For local Ollama models:
LOCAL_TASK_MODEL_<NAME>=ollama_chat/<ollama-model-name>
```

### 2. Config class

- **Bedrock model** → add a field to `aws_config.py` (`AWSConfig`)
- **Ollama model** → add a field to `ollama_config.py` (`OllamaConfig`)

### 3. LLM provider factory

In `llm_provider.py`:

1. Add the model name to the `TaskModelName` type alias
2. Add a `case` branch in `get_task_lm()` with the correct provider routing

### 4. Verify

Run the test notebook (`scripts/notebooks/test_ollama.ipynb`) or unit
tests (`tests/unit/test_config.py`) to confirm the new model loads correctly.
