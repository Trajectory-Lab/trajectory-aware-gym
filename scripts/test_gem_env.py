# %% [markdown]
# # Test GEM Enviroment
#
# This notebook runs a very simple Agent to play a number guessing game. The Agent uses binary search to gues sthe number between 1 and 10 (inclusive).

# %%
import re
import gem
from typing import Optional

# %% [markdown]
# ## Class Instantiation


# %%
class SimpleGuessAgent:
    def __init__(self) -> None:
        self.low: Optional[int] = None
        self.high: Optional[int] = None
        self.last_action: Optional[int] = None

    def parse_range(self, observation: str) -> None:
        numbers = list(map(int, re.findall(r"\d+", observation)))
        if len(numbers) >= 2:
            self.low, self.high = numbers[0], numbers[1]

    def act(self, observation: Optional[str]) -> str:
        # First step: read range
        if observation is not None and (self.low is None or self.high is None):
            self.parse_range(observation)

        # Update bounds from feedback
        if observation is not None and self.last_action is not None:
            if "higher" in observation:
                self.low = self.last_action + 1
            elif "lower" in observation:
                self.high = self.last_action - 1

        # Safety check
        if self.low is None or self.high is None:
            raise ValueError("Range not initialized from observation.")

        # Binary search guess
        self.last_action = (self.low + self.high) // 2
        return f"\\boxed{{{self.last_action}}}"

    def learn(self, observation: Optional[str], action: str, reward: float) -> None:
        pass


# %% [markdown]
# ## Train the Agent

# %%
# Instantiate the GEM environment
env = gem.make("game:GuessTheNumber-v0-easy")
observation, info = env.reset()

print("Initial observation:", observation)

agent = SimpleGuessAgent()

turn = 1

while True:
    action = agent.act(observation)
    next_obs, reward, terminated, truncated, info = env.step(action)

    # Extract number from \boxed{}
    guess = action.replace("\\boxed{", "").replace("}", "")

    print(f"\nTurn {turn}")
    print(f"=" * 60)
    print(f"Agent guess: {guess}")
    print(f"Environment: {next_obs}")
    print(f"Reward: {reward}")

    observation = next_obs
    turn += 1

    if terminated or truncated:
        print("\nEpisode finished.")
        break
