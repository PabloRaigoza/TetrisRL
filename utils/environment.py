from tetris_gymnasium.mappings.rewards import RewardsMapping
from tetris_gymnasium.envs import *
import gymnasium as gym


# Constants
MAX_STEPS = 1000


# Custom reward mapping
rewards_mapping = RewardsMapping()
rewards_mapping.game_over = -100
rewards_mapping.invalid_action = -10


# BC Data Collection
def makeBC():
    return gym.make(
        "tetris_gymnasium/Tetris",
        render_mode="human",
        render_upscale=30,
        rewards_mapping=rewards_mapping
    )

makeBC()
