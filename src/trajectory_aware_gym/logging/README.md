# GEM Episode Logging

This module provides trajectory logging for GEM episodes with Pydantic schema validation.

## Features

- **Schema-validated logging**: Pydantic models ensure data integrity
- **Per-step tracking**: Captures observations, actions, rewards, and metadata
- **JSON storage**: Human-readable logs with ISO 8601 timestamps
- **Episode metadata**: Tracks success, total reward, and step count

## Quick Start

### Running a Single Episode

```bash
# Run the example script
uv run python scripts/run_gem_episode.py
```

This will:
1. Execute one complete episode of GuessTheNumber-v0-easy
2. Log the full trajectory to `logs/`
3. Display progress and summary statistics
4. Validate the log schema

### Programmatic Usage

```python
import gem
from trajectory_aware_gym.logging import TrajectoryLogger

# Initialize environment and logger
env = gem.make("game:GuessTheNumber-v0-easy")
logger = TrajectoryLogger(log_dir="logs")

# Start episode
episode_id = logger.start_episode(env.spec.id)
observation, info = env.reset()

# Run episode
while True:
    action = agent.act(observation)
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    # Log this step
    logger.log_step(
        step=step,
        observation=observation,
        action=action,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )
    
    observation = next_obs
    if terminated or truncated:
        break

# Save trajectory
log_file = logger.save_episode()
print(f"Saved to: {log_file}")
```

### Loading and Validating Logs

```python
from trajectory_aware_gym.logging import TrajectoryLogger

logger = TrajectoryLogger()
episode = logger.load_episode("logs/episode_123.json")

# Access trajectory data
print(f"Environment: {episode.env_id}")
print(f"Steps: {episode.num_steps}")
print(f"Success: {episode.success}")
print(f"Total Reward: {episode.total_reward}")

# Iterate through steps
for step in episode.steps:
    print(f"Step {step.step}: {step.action} -> {step.reward}")
```

## Log Schema

### EpisodeLog

```json
{
  "episode_id": "game:GuessTheNumber-v0-easy_20260213_173351_762502",
  "env_id": "game:GuessTheNumber-v0-easy",
  "timestamp": "2026-02-13T17:33:51.762502+00:00",
  "steps": [...],
  "total_reward": 1.0,
  "success": true,
  "num_steps": 4
}
```

### TrajectoryStep

```json
{
  "step": 0,
  "observation": "You are playing Guess The Number...",
  "action": "\\boxed{5}",
  "reward": 0.0,
  "terminated": false,
  "truncated": false,
  "info": {"suffix": "Enter your next guess."}
}
```

## Testing

```bash
# Run unit tests
uv run pytest tests/unit/test_trajectory_logger.py -v

# Run integration tests
uv run pytest tests/integration/test_gem_episode_execution.py -v

# Run all tests with coverage
uv run pytest tests/ --cov=src/trajectory_aware_gym/logging
```

## Implementation Notes

- Logs are saved to `logs/` directory by default (configurable)
- Episode IDs include timestamp for uniqueness: `{env_id}_{YYYYMMDD_HHMMSS_microseconds}`
- Schema validation happens on both save and load operations
- Success is determined by positive reward on termination
- Total reward accumulates across all steps

## Use Cases

1. **Training Data Collection**: Capture agent behavior for analysis
2. **Debugging**: Inspect full trajectories for failed episodes
3. **Evaluation**: Store test episode results with metadata
4. **Reproducibility**: Archive exact action sequences for replay
5. **Research**: Analyze trajectory patterns across multiple runs
