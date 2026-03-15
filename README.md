# trajectory-aware-gym

Trajectory-aware prompt optimization for agentic LLMs. Compares **token-space prompt optimization** (GEPA) against **weight-space reinforcement learning** (PPO, GRPO) on GEM environments.

## Overview

This Harvard Extension School capstone project evaluates whether evolutionary prompt optimization can match RL performance at a fraction of the computational cost. The core contribution is the **GEM-DSPy adapter**, which bridges OpenAI Gym-style environments with DSPy's prompt optimization framework.

**Key Features:**
- 🔄 **GEM-DSPy Adapter**: Seamless integration between GEM environments and DSPy modules
- 📊 **Trajectory-Aware Fitness**: Per-turn reward signals for fine-grained credit assignment
- 🧬 **GEPA Integration**: Evolutionary prompt optimization with a Bedrock-hosted reflection model
- 💰 **Cost Tracking**: Built-in token and cost tracking for all LLM calls
- 🎯 **Two Published-Baseline Environments**: Orz57K and HotpotQA, with `quick-test` for fast iteration

## Current Status

K4 is functionally complete: the repository now supports an end-to-end GEPA + DSPy + GEM dry-run that:
- samples real `math:Orz57K` tasks
- executes GEM episodes through the concrete adapter
- scores trajectories with the DSPy-compatible fitness metric
- runs `dspy.GEPA.compile()` and saves optimizer artifacts under `logs/gepa-dry-run/`

What this does **not** yet prove:
- reliable prompt improvement over the seed instructions
- robust multi-step / tool-using behavior in the dry-run setting
- experiment-ready evaluation quality for the full capstone hypotheses

## Prerequisites

### Clone the repository

```bash
git clone https://github.com/Trajectory-Lab/trajectory-aware-gym.git
cd trajectory-aware-gym
```

### Install uv

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or install via package managers:
- macOS: `brew install uv`
- Linux: `pip install uv`
- Windows: `pip install uv`

## Setup

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Run setup** (install hooks + verify):
   ```bash
   poe setup
   ```

3. **Configure credentials** (copy and edit):
   ```bash
   cp .env.example .env
   # Add your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
   ```

   Non-sensitive defaults live in [`trajectory-aware-gym.yaml`](src/trajectory_aware_gym/config/trajectory-aware-gym.yaml). Env vars in `.env` override YAML values. See [docs/configuration.md](docs/configuration.md) for details.

## Quickstart

**Minimal real GEPA + GEM dry-run (Orz57K):**

```bash
# Run the bounded end-to-end dry-run
poe dry-run-gepa --fresh
```

This uses [`experiments/math-dry-run/config.yaml`](experiments/math-dry-run/config.yaml):
- `math:Orz57K`
- `bedrock/us.meta.llama3-1-8b-instruct-v1:0` as the task model
- `bedrock/openai.gpt-oss-120b-1:0` as the GEPA reflection model
- `12` bounded GEPA metric calls
- `5` seeded training examples / `2` validation examples
- `2` max environment steps
- `2048` max response tokens per call

The dry-run currently validates the end-to-end integration and artifact generation. It does not yet guarantee multi-step tool use on Orz57K; recent runs have mostly produced single-step trajectories without executed tool calls.

**Test your setup with the interactive notebook:**

Open [test_bedrock.ipynb](scripts/notebooks/test_bedrock.ipynb) to verify your AWS Bedrock configuration and explore:
- Basic LiteLLM calls with Bedrock models
- Cost and token tracking
- DSPy integration with LiteLLM
- Temperature effects on model outputs
- Multi-model comparisons

The notebook includes examples of the available Bedrock Llama task models and demonstrates how to track costs for both direct LiteLLM calls and DSPy workflows.

**Run the notebook:**
```bash
# Start Jupyter
uv run jupyter lab

# Or use VSCode's built-in notebook support
# Just open scripts/notebooks/test_bedrock.ipynb in VSCode
```

## Project Structure

```
src/trajectory_aware_gym/
├── adapters/    # GEM-DSPy integration
├── fitness/     # Trajectory-aware fitness functions
├── optimizers/  # GEPA configuration
├── config/      # Centralized settings (YAML + env var loading)
└── utils/       # Utilities
```

## Development

### Branching Strategy

**Always branch off `development` (not `main`):**

```bash
# Start new feature
git checkout development
git pull origin development
git checkout -b feat/your-feature-name

# Make your changes and commit
git add .
git commit -m "feat(scope): description"

# Push and create PR to development
git push -u origin feat/your-feature-name
# Create PR via GitHub UI targeting development branch
```

**Important:** Never merge directly to `main`. The `main` branch is production-equivalent and only updated at release milestones.

### Commands

```bash
# Run tests
poe test

# Run tests with coverage
poe test-cov

# Format code
poe format

# Lint
poe lint

# Type check
poe typecheck
```

## Documentation

- **[docs/configuration.md](docs/configuration.md)**: Configuration guide — YAML schema, env var overrides, adding models
- **[AGENTS.md](AGENTS.md)**: Comprehensive guidelines for AI coding assistants working on this project
- **[.env.example](.env.example)**: Example environment configuration with AWS Bedrock settings
- **[integration_test_matrix.md](docs/integration_test_matrix.md)**: Integration test plan and CI matrix for Phase 2
- **[phase2_handoff.md](docs/phase2_handoff.md)**: Unit/integration testing handoff summary
- **[test_bedrock.ipynb](scripts/notebooks/test_bedrock.ipynb)**: Interactive notebook for testing setup and exploring LLM integrations

## License

MIT
