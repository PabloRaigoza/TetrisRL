from tetris_gymnasium.wrappers.observation import FeatureVectorObservation
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.mappings.rewards import RewardsMapping
from tetris_gymnasium.envs import Tetris

import gymnasium as gym
import numpy as np
import copy

from typing import Any


# Constants
MAX_STEPS = 1000


# Custom reward mapping
rewards_mapping = RewardsMapping()
rewards_mapping.game_over = -100
rewards_mapping.invalid_action = -10


# Make Unwrapped Tetris environment
def makeStandard():
    return gym.make(
        "tetris_gymnasium/Tetris",
        render_mode="human",
        render_upscale=40,
        rewards_mapping=rewards_mapping
    )


# Make Grouped Tetris environment
def makeGrouped():
    env = Tetris(render_mode="human", render_upscale=40, rewards_mapping=rewards_mapping)
    return GroupedActionsObservations(env=env)
