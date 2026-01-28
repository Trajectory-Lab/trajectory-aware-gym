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
- Never mention "Claude/Codex/Cursor/etc" in commit messages unless directly relevant

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
   - Direct commits to `main` are not allowed

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
- Configuration via Pydantic Settings with `.env` file

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

### Environment Variables (.env)

Critical settings:
- **AWS_REGION**, **AWS_ACCESS_KEY_ID**, **AWS_SECRET_ACCESS_KEY**: Bedrock access
- **BEDROCK_QWEN3_1_7B**, **BEDROCK_CLAUDE_SONNET_4_5**: Model IDs
- **GEPA_BUDGET**: light/medium/heavy (controls iterations and population)
- **GEM_TEMPERATURE_TRAIN**: 1.0 for exploration, **GEM_TEMPERATURE_EVAL**: 0.0 for determinism

### Programmatic Access

```python
from trajectory_aware_gym.config import AWSConfig, GEPAConfig, GEMConfig

aws = AWSConfig()  # Auto-loads from .env
gepa = GEPAConfig()
gem = GEMConfig()
```

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
- **Do** log all API calls with token counts for precise cost tracking
