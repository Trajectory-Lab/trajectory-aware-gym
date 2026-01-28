# trajectory-aware-gym

Trajectory-aware prompt optimization for agentic LLMs. Benchmarks GEPA against RL baselines on GEM environments.

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

3. **Configure AWS** (copy and edit):
   ```bash
   cp .env.example .env
   ```

## Project Structure

```
src/trajectory_aware_gym/
├── adapters/    # GEM-DSPy integration
├── fitness/     # Trajectory-aware fitness functions
├── optimizers/  # GEPA configuration
├── config/      # Settings and AWS configuration
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

## License

MIT
