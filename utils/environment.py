from tetris_gymnasium.mappings.rewards import RewardsMapping
from tetris_gymnasium.envs import *
import gymnasium as gym
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation


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
        render_upscale=40,
        rewards_mapping=rewards_mapping
    )


# DA Data Collection
def makeDA():
    return gym.make(
        "tetris_gymnasium/Tetris",
        render_mode="human",
        render_upscale=40,
        rewards_mapping=rewards_mapping
    )

import copy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

from tetris_gymnasium.components.tetromino import Tetromino
from tetris_gymnasium.envs import Tetris

class TetrisCopy(Tetris):
    def _get_obs(self) -> "dict[str, Any]":
        """Return the current board as an observation."""
        # Include the active tetromino on the board for the observation.
        board_obs = self.project_tetromino()

        # Create a mask where the active tetromino is
        active_tetromino_slices = self.get_tetromino_slices(
            self.active_tetromino, self.x, self.y
        )
        active_tetromino_mask = np.zeros_like(board_obs)
        # active_tetromino_mask[active_tetromino_slices] = self.active_tetromino.matrix
        active_tetromino_mask[active_tetromino_slices] = 1
        active_tetromino = (self.active_tetromino.matrix.astype(np.uint8), self.x, self.y)

        # Holder
        max_size = self.padding
        holder_tetrominoes = self.holder.get_tetrominoes()
        if len(holder_tetrominoes) > 0:
            # Pad all tetrominoes to be the same size
            for index, t in enumerate(holder_tetrominoes):
                holder_tetrominoes[index] = np.pad(
                    t.matrix,
                    (
                        (0, max_size - t.matrix.shape[0]),
                        (0, max_size - t.matrix.shape[1]),
                    ),
                )
            # Concatenate all tetrominoes horizontally
            holder_obs = np.hstack(holder_tetrominoes)
        else:
            holder_obs = np.ones((max_size, max_size * self.holder.size))

        # Queue
        queue_tetrominoes = self.queue.get_queue()
        for index, t_id in enumerate(queue_tetrominoes):
            # Pad all tetrominoes to be the same size
            t = copy.deepcopy(self.tetrominoes[t_id])
            t.matrix = np.pad(
                t.matrix,
                ((0, max_size - t.matrix.shape[0]), (0, max_size - t.matrix.shape[1])),
            )
            # Safe padded result back to the array
            queue_tetrominoes[index] = t.matrix
        # Concatenate all tetrominoes horizontally
        queue_obs = np.hstack(queue_tetrominoes)

        return {
            "board": board_obs.astype(np.uint8),
            "active_tetromino_mask": active_tetromino_mask.astype(np.uint8),
            "active_tetromino": active_tetromino,
            "holder": holder_obs.astype(np.uint8),
            "queue": queue_obs.astype(np.uint8),
        }

class GroupedActionsObservationsCopy(GroupedActionsObservations):
    def step(self, action):
        """Performs the action.

        Args:
            action: The action to perform.

        Returns:
            The observation, reward, game over, truncated, and info.
        """
        x, r = self.decode_action(action)

        if self.legal_actions_mask[action] == 0:
            if self.terminate_on_illegal_action:
                observation = (
                    np.ones(self.observation_space.shape) * self.observation_space.high
                )
                game_over, truncated = True, False
                info = {"action_mask": self.legal_actions_mask, "lines_cleared": 0}
            else:
                (
                    observation,
                    reward,
                    game_over,
                    truncated,
                    info,
                ) = self.env.unwrapped.step(self.env.unwrapped.actions.no_op)
                observation = self.observation(observation)
                info["action_mask"] = self.legal_actions_mask

            reward = self.env.unwrapped.rewards.invalid_action
            return observation, reward, game_over, truncated, info

        new_tetromino = copy.deepcopy(self.env.unwrapped.active_tetromino)

        # Set new x position
        x += self.env.unwrapped.padding

        # Set new rotation
        for _ in range(r):
            new_tetromino = self.env.unwrapped.rotate(new_tetromino)

        # Apply rotation and movement (x,y)
        self.env.unwrapped.x = x
        self.env.unwrapped.active_tetromino = new_tetromino

        # Perform the action
        observation, reward, game_over, truncated, info = self.env.unwrapped.step(
            self.env.unwrapped.actions.hard_drop
        )
        board = observation
        if self.observation_wrappers:
            for wrapper in self.observation_wrappers:
                board = wrapper.observation(board)
        info["board"] = board
        info["original_board"] = observation["board"]
        info["active_tetromino"] = observation["active_tetromino"]


        observation = self.observation(observation)  # generates legal_action_mask
        info["action_mask"] = self.legal_actions_mask

        return observation, reward, game_over, truncated, info
    def reset(
        self, *, seed: "int | None" = None, options: "dict[str, Any] | None" = None
    ) -> "tuple[dict[str, Any], dict[str, Any]]":
        """Resets the environment.

        Args:
            seed: The seed to use for the random number generator.
            options: The options to use for the environment.

        Returns:
            The observation and info.
        """
        self.legal_actions_mask = np.ones(self.action_space.n)
        observation, info = self.env.reset(seed=seed, options=options)
        board = observation
        if self.observation_wrappers:
            for wrapper in self.observation_wrappers:
                board = wrapper.observation(board)
        info["board"] = board
        info["original_board"] = observation["board"]
        info["active_tetromino"] = observation["active_tetromino"]

        observation = self.observation(observation)  # generates legal_action_mask
        info["action_mask"] = self.legal_actions_mask

        return observation, info
        


def makeGroupedActionsWrapper():
    env=makeDA()
    # env = TetrisCopy(
    #     render_mode="human",
    #     render_upscale=40,
    #     rewards_mapping=rewards_mapping
    # )
    # groupedActionsWrapper = GroupedActionsObservationsCopy(
    #     env=env,
    #     observation_wrappers=[FeatureVectorObservation(env)]
    # )
    groupedActionsWrapper = GroupedActionsObservations(
        env=env,
        observation_wrappers=[FeatureVectorObservation(env)]
    )
    return groupedActionsWrapper
