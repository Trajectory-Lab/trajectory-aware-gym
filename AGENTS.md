# AGENTS.md

**This is the source of truth for all AI coding agents working in this repository.**

This file provides comprehensive guidance for AI coding assistants (Claude Code, Cursor, Copilot, etc.) when working with code in this repository.

## Project Overview

This is a Harvard Extension School capstone project comparing **token-space prompt optimization** (GEPA) against **weight-space reinforcement learning** (PPO, GRPO) for agentic LLM tasks. The core contribution is the **GEM-DSPy adapter**, which bridges OpenAI Gym-style RL environments with prompt optimization frameworks.

## Package Management

**Use `uv` for all Python operations:**
- Run scripts: `uv run python script.py`
- Add dependencies: `uv add package_name`
- Sync dependencies: `uv sync`

**Use `poe` task runner for common commands:**
- Setup (hooks + verify): `poe setup`
- Run tests: `poe test`
- Run tests with coverage: `poe test-cov`
- Format code: `poe format`
- Lint code: `poe lint`
- Type check: `poe typecheck`

## Version Control

- This project uses **`git`** (not `glab`)
- Follow conventional commits:
  ```
  <type>(<scope>): <description>

  Examples:
  feat(adapter): add GEM environment wrapper for DSPy
  fix(fitness): correct per-turn reward accumulation
  refactor(config): consolidate AWS settings
  ```
- Never mention "Generated with Claude/Codex/Cursor/etc" in commit/PR messages unless directly relevant

### Branching Strategy

**Branch Structure:**
- `main`: Production-equivalent branch for versioning. Represents stable releases.
- `development`: Active development branch. All feature work merges here.
- `feat/*`, `fix/*`, `refactor/*`: Feature branches for specific work.

**Development Flow:**
1. Create feature branches **off `development`** (not `main`):
   ```bash
   git checkout development
   git pull origin development
   git checkout -b feat/your-feature-name
   ```

2. Work on your feature with regular commits

3. When done, create a PR to merge back to `development`:
   ```bash
   git push -u origin feat/your-feature-name
   # Then create PR via GitHub UI targeting development branch
   ```

4. **Never merge directly to `main`**:
   - `main` is cut from `development` at release milestones
   - Only project maintainers merge `development` → `main` for versioned releases
   - Direct commits to `main` and `development` are blocked by GitHub Actions
   - All changes must go through pull requests

## Architecture Overview

### Core Paradigm

The project evaluates two optimization paradigms:

1. **Weight-Space (RL)**: PPO/GRPO modifying model parameters via policy gradients
2. **Token-Space (Prompt Optimization)**: GEPA evolving system prompts via evolutionary search

### Key Components

**GEM-DSPy Adapter** (`src/trajectory_aware_gym/adapters/`):
- Bridges GEM's OpenAI Gym interface (`reset()`, `step()`) with DSPy's module system
- Captures full trajectory traces: `(s₀, a₀, r₁, s₁, a₁, r₂, ..., sₜ)`
- Routes tool calls (Python exec, web search, shell) through GEM's standardized interfaces
- Converts environment observations to LLM-consumable text format

**Trajectory-Aware Fitness** (`src/trajectory_aware_gym/fitness/`):
- Leverages GEM's **per-turn rewards** for fine-grained credit assignment
- Fitness function: `F(τ) = Σ γ^(T-t) · wₜ · Rₜ + λ Σ rₜ`
  - `γ`: discount factor (reverse-time weighting, final steps get full credit)
  - `wₜ`: per-turn weight
  - `Rₜ`: final reward indicator (1 for success, 0 for failure)
  - `rₜ`: auxiliary per-turn rewards (syntactic checks, loop detection)
- Composite metrics include loop detection penalty and step efficiency bonus

**GEPA Integration** (`src/trajectory_aware_gym/optimizers/`):
- Uses Claude Sonnet 4.5 as reflection model (distinct from task model)
- Maintains **Pareto frontier** of prompts (not single best) to preserve diversity
- Budget modes: light/medium/heavy controlling iteration count and population size

**AWS/LLM Infrastructure** (`src/trajectory_aware_gym/config/`):
- All LLM calls route through **LiteLLM** for unified provider interface
- Task models: Qwen3 (1.7B, 4B) via AWS Bedrock for fair comparison with GEM's RL baselines
- Reflection model: Claude Sonnet 4.5 via Bedrock for GEPA mutations
- Configuration via YAML (`src/trajectory_aware_gym/config/trajectory-aware-gym.yaml`) with `.env` overrides (see Configuration Management)

### Environments

Three GEM environments from the proposal:
1. **Math12K**: Chain-of-thought mathematical reasoning (single-turn)
2. **CodeContest**: Competitive programming with test execution (multi-turn, tool-using)
3. **HotpotQA**: Multi-hop question answering (multi-turn, retrieval-heavy)

## Development Commands

### Testing
```bash
# Run all tests
poe test

# Run with coverage
poe test-cov

# Run specific test file
uv run pytest tests/unit/test_fitness_functions.py -v

# Run integration tests only
uv run pytest tests/integration/ -v
```

**Unit Test Guidelines:**
- Use `pytest.mark.parametrize` for testing multiple inputs/scenarios
- Parameters should be **comprehensive** and include edge cases (empty strings, zero, negative values, boundary conditions, None where applicable)
- Prefer parametrized tests over repetitive test methods
- Use fixtures for shared setup (e.g., `tmp_path`, `monkeypatch`)
- For config tests, use `Settings.reset()` in fixtures to clear singleton state between tests
- Tests load from the production `src/trajectory_aware_gym/config/trajectory-aware-gym.yaml`; use `monkeypatch.setenv()` for overrides

```python
# Good: comprehensive parametrized test with edge cases
@pytest.mark.parametrize(
    ("input_val", "expected"),  # Use tuple, not comma-separated string
    [
        (0, 0),           # zero
        (1, 1),           # minimum positive
        (-1, 1),          # negative
        (100, 100),       # typical value
        (float("inf"), float("inf")),  # edge case
    ],
)
def test_absolute_value(input_val, expected):
    assert abs(input_val) == expected
```

### Code Quality

**Pre-commit hooks are configured and will run automatically on `git commit`**

```bash
# Run all pre-commit hooks manually
uv run pre-commit run --all-files

# Install pre-commit hooks (if not already done)
uv run pre-commit install

# Update hook versions
uv run pre-commit autoupdate
```

**Manual code quality checks:**

```bash
# Format code
poe format

# Lint
poe lint

# Type check
poe typecheck

# Security scanning
uv run bandit -r src
```

**Pre-commit hooks include:**
- Black (formatting)
- Ruff (linting + auto-fix)
- Pyright (type checking, excluding tests/scripts/examples)
- Bandit (security scanning, excluding tests)
- Detect-secrets (credential scanning)
- Standard checks (trailing whitespace, YAML/TOML/JSON validation, etc.)

### Running Experiments
```bash
# Activate environment first (if needed for interactive work)
source .venv/bin/activate

# Run GEPA optimization on Math12K
uv run python examples/run_gepa_math12k.py

# Compare against RL baselines
uv run python scripts/compare_baselines.py --environment math12k

# Run with specific configuration
uv run python scripts/run_experiment.py \
  --config experiments/configs/baseline.yaml \
  --replications 3
```

## Configuration Management

Configuration is centralized in `src/trajectory_aware_gym/config/trajectory-aware-gym.yaml` with `.env` overrides.

**Priority (highest → lowest):** `.env` / env vars → `src/trajectory_aware_gym/config/trajectory-aware-gym.yaml`

No defaults are hardcoded in Python — all values come from `.env` or YAML.

### Key Files

- **`src/trajectory_aware_gym/config/trajectory-aware-gym.yaml`**: Non-sensitive defaults (checked into git)
- **`.env`**: Secrets and per-developer overrides (git-ignored)
- **`src/trajectory_aware_gym/config/core.py`**: Settings loader + sub-models

### Environment Variables (.env)

Env vars use `PREFIX_FIELD` naming (e.g., `AWS_REGION`, `GEM_MAX_STEPS`):
- **AWS_ACCESS_KEY_ID**, **AWS_SECRET_ACCESS_KEY**: Bedrock credentials
- **AWS_REGION**: AWS region override
- **GEPA_BUDGET**: light/medium/heavy (controls iterations and population)
- **GEM_MAX_STEPS**: Max steps per episode
- **GEM_TEMPERATURE_TRAIN**: 1.0 for exploration, **GEM_TEMPERATURE_EVAL**: 0.0 for determinism

Any YAML value can be overridden via env var using its section prefix:
`{SECTION_PREFIX}_{FIELD_NAME}` (e.g., `OLLAMA_API_BASE`, `LOG_LEVEL`)

### Programmatic Access

```python
from trajectory_aware_gym.config import settings

# Access sub-configs via properties
settings.aws.region
settings.gem.max_steps
settings.gepa.budget
settings.ollama.api_base
```

## Cost and Token Tracking

**Always track and report cost and token usage when working with LLMs.** This is critical for:
- Budget management and cost control
- Comparing efficiency across approaches (key for H2 hypothesis)
- Reproducing experiments with precise resource accounting
- Identifying optimization opportunities

### Implementation Guidelines

**For Direct LiteLLM Calls:**
```python
from litellm import completion, completion_cost

response = completion(
    model="bedrock/model-id",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=100
)

# Track immediately after call
tokens = response.usage.total_tokens
cost = completion_cost(completion_response=response)
print(f"Tokens: {tokens}, Cost: ${cost:.6f}")
```

**For DSPy Calls:**
```python
import dspy
from dspy import LM
from litellm import completion_cost

lm = LM(model="bedrock/model-id", temperature=0, max_tokens=100)
dspy.configure(lm=lm)

# Make DSPy call
predictor = dspy.Predict(SomeSignature)
result = predictor(input="...")

# Access call history for cost tracking
last_call = lm.history[-1]
tokens = last_call['usage']['total_tokens']

if 'response' in last_call:
    cost = completion_cost(completion_response=last_call['response'])
    print(f"Tokens: {tokens}, Cost: ${cost:.6f}")
```

**Best Practices:**
- Track costs in notebooks, scripts, and experiments
- Log token usage alongside performance metrics
- Accumulate total costs for multi-step operations
- Include cost reports in experiment outputs
- Use cost tracking to validate budget estimates from methodology

### Why This Matters

Per H2, we hypothesize GEPA requires ≥1 order of magnitude fewer resources than RL. **Precise cost tracking is essential evidence for this claim.** Every LLM call should be accounted for.

## Experimental Design Notes

### Cost Estimation (from Methodology)

Per experiment with 3 replications:
- Training: ~27M tokens
- Validation: ~27M tokens
- Evaluation: ~1.2M tokens
- Reflection: ~540K tokens
- **Total**: ~94M tokens × 3 = ~283M tokens

### Evaluation Protocol

Match GEM's RL settings exactly:
- Max response length: 4096 tokens/turn
- Training temp: 1.0, Eval temp: 0.0
- Top-p: 1.0, top-k: disabled
- Use identical held-out test sets
- 5 independent rollouts per test task (different seeds)

### Statistical Analysis

- Use **TOST (Two One-Sided Tests)** for equivalence testing (not superiority)
- Equivalence margin: ±5 percentage points
- α = 0.05
- Report: point estimates, 95% CIs (bootstrap), effect sizes (Cohen's d)

## Code Style & Clean Code Principles

### Core Principles

**KISS (Keep It Simple, Stupid)**
- Don't build a rocket ship when a bicycle will do
- Prefer straightforward solutions over clever complexity
- Example: Extract complex boolean logic into well-named functions

```python
# Don't do this
if user.roles and 'admin' in user.roles and user.permissions and any(p.code == 'ALL_ACCESS' for p in user.permissions):
    ...

# Do this
def is_admin(user):
    return 'admin' in user.roles and has_all_access(user)

if is_admin(user):
    ...
```

**DRY (Don't Repeat Yourself)**
- Consolidate repeated logic, but don't over-abstract prematurely
- For this project: shared trajectory processing, fitness calculations, AWS client setup
- Every piece of knowledge should have a single source of truth

**YAGNI (You Ain't Gonna Need It)**
- Don't write code until you absolutely need it
- Build only what the research requires; refactor when needs arise
- For this project: Don't add "future-proof" features for environments we're not testing

### Project-Specific Guidelines

- Minimize comments; prefer self-documenting code with clear names
- Comment only for non-obvious logic (e.g., fitness function math, trajectory weighting)
- Use type hints for function signatures
- Follow Pydantic models for all configuration classes
- Keep adapters simple: focus on observation/action translation, not business logic
- **Always track and include cost/token usage** when making LLM calls (see "Cost and Token Tracking" section)

### Modern Python 3.13+ Syntax

- Use modern type hints: `list[str]`, `dict[str, int]`, `int | None` (not `List`, `Dict`, `Optional`)
- Use `type` for type aliases: `type TrajectoryStep = dict[str, Any]`
- Use pattern matching (`match`/`case`) for complex conditionals
- Use f-strings exclusively, prefer `f"{value=}"` for debugging

## Project Timeline (16 weeks)

- **Weeks 1-2** (Phase 1): Environment setup ← **Currently here**
- **Weeks 3-7** (Phase 2): GEM-DSPy-GEPA integration
- **Weeks 8-11** (Phase 3): Primary experiments
- **Weeks 12-13** (Phase 4): Generalization & ablations
- **Weeks 14-16** (Phase 5): Analysis & writing

## Key Research Hypotheses

**H1 (Performance)**: GEPA achieves task success within 5pp of RL baseline (equivalence, not superiority)
**H2 (Compute)**: GEPA requires ≥1 order of magnitude fewer resources than RL training
**H3 (Mechanism)**: Composite fitness (loop detection, step efficiency) significantly improves convergence

## Important Constraints

- **Do NOT** train RL agents ourselves; compare against published GEM baselines
- **Do NOT** use test set during optimization (strict held-out evaluation)
- **Do** maintain identical evaluation protocols between paradigms for fair comparison
- **Do** track and report cost/token usage for ALL LLM calls (critical for H2 hypothesis validation)

## Documentation Maintenance

When adding new features or deprecating existing ones, update the relevant documentation:

- **[README.md](README.md)**: Project overview, setup instructions, project structure
- **[docs/02-architecture/configuration.md](docs/02-architecture/configuration.md)**: Configuration schema, env var conventions, adding models/sections
- **[AGENTS.md](AGENTS.md)**: This file — agent guidelines, architecture, development commands
- **[.env.example](.env.example)**: Keep in sync with any new secret/override env vars

Do not let documentation drift from the code. If you change a config field, model name, or access pattern, update the corresponding docs in the same PR.
