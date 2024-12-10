from tetris_gymnasium.wrappers.observation import FeatureVectorObservation
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.mappings.rewards import RewardsMapping

import gymnasium as gym
import numpy as np
import copy


# Custom reward mapping
MAX_STEPS = 1000
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


# Make Grouped Actions Wrapper
class GroupedActionsObservationsCopy(GroupedActionsObservations):
    def observation(self, observation):
        board_obs = observation["board"]
        holder_obs = observation["holder"]
        queue_obs = observation["queue"]

        self.legal_actions_mask = np.ones(self.action_space.n)
        grouped_board_obs = []
        if self.env.unwrapped.game_over: # Game over (previous step)
            np.zeros(self.observation_space.shape)


        t = self.env.unwrapped.active_tetromino
        for x in range(self.env.unwrapped.width):
            x = self.env.unwrapped.padding + x
            x -= 1 if self.env.unwrapped.active_tetromino.id == 2 else 0

            for r in range(4):
                y = 0
                if r > 0: t = self.env.unwrapped.rotate(t)
                while not self.env.unwrapped.collision(t, x, y + 1): y += 1

                if self.collision_with_frame(t, x, y):
                    self.legal_actions_mask[self.encode_action(x - self.env.unwrapped.padding, r)] = 0
                    grouped_board_obs.append(np.ones_like(board_obs))

                elif self.env.unwrapped.collision(t, x, y):
                    grouped_board_obs.append(np.zeros_like(board_obs))

                else:
                    grouped_board_obs.append(self.env.unwrapped.project_tetromino(t, x, y))

            t = self.env.unwrapped.rotate(t)  # Reset rotation


        # Apply wrappers
        if self.observation_wrappers is not None:
            for i, observation in enumerate(grouped_board_obs):
                # Recreate the original environment observation
                grouped_board_obs[i] = {
                    "board": observation,
                    "active_tetromino_mask": np.zeros_like(observation),
                    "holder": holder_obs,
                    "queue": queue_obs,
                }

                # Validate that observations are equal
                assert (grouped_board_obs[i].keys() == self.env.unwrapped.observation_space.keys())

                # Apply wrappers to all the original observations
                for wrapper in self.observation_wrappers:
                    grouped_board_obs[i] = wrapper.observation(grouped_board_obs[i])

        grouped_board_obs = np.array(grouped_board_obs)
        return grouped_board_obs


    def step(self, action):
        x, r = self.decode_action(action)
        x -= 1 if self.env.unwrapped.active_tetromino.id == 2 else 0


        if self.legal_actions_mask[action] == 0:
            if self.terminate_on_illegal_action:
                observation = (np.ones(self.observation_space.shape) * self.observation_space.high)
                game_over, truncated = True, False
                info = {"action_mask": self.legal_actions_mask, "lines_cleared": 0}

            else:
                (observation, reward, game_over, truncated, info,) \
                    = self.env.unwrapped.step(self.env.unwrapped.actions.no_op)
                observation = self.observation(observation)
                info["action_mask"] = self.legal_actions_mask

            reward = self.env.unwrapped.rewards.invalid_action
            return observation, reward, game_over, truncated, info

        new_tetromino = copy.deepcopy(self.env.unwrapped.active_tetromino)
        x += self.env.unwrapped.padding
        for _ in range(r):
            new_tetromino = self.env.unwrapped.rotate(new_tetromino)

        # Apply rotation and movement (x, y)
        self.env.unwrapped.x = x
        self.env.unwrapped.active_tetromino = new_tetromino

        # Perform the action
        observation, reward, game_over, truncated, info \
            = self.env.unwrapped.step(self.env.unwrapped.actions.hard_drop)
        board = observation
        if self.observation_wrappers:
            for wrapper in self.observation_wrappers:
                board = wrapper.observation(board)
        info["board"] = board

        observation = self.observation(observation)  # Generates legal_action_mask
        info["action_mask"] = self.legal_actions_mask
        return observation, reward, game_over, truncated, info


# Make Heuristic Feature Wrapper
class FeatureVectorObservationCopy(FeatureVectorObservation):
    def calc_lines(self, board):
        return sum([not np.any(board[r] == 0) for r in range(board.shape[0])])


    def column_height(self, board, column):
        r = 0
        while r < board.shape[0] and board[r][column] == 0: r += 1
        return board.shape[0] - r


    def aggregate_height(self, board):
        return sum([self.column_height(board, c) for c in range(board.shape[1])])


    def holes(self, board):
        count = 0
        for c in range(board.shape[1]):
            hole = False
            for r in range(board.shape[0]):
                if board[r][c] != 0: hole = True
                elif board[r][c] == 0 and hole: count += 1
        return count


    def calc_bumpiness(self, board):
        return sum([abs(self.column_height(board, c) - self.column_height(board, c + 1)) for c in range(board.shape[1] - 1)])


    def observation(self, observation):
        # Board
        board_obs = observation["board"]
        active_tetromino_mask = observation["active_tetromino_mask"]
        board_obs[active_tetromino_mask] = 0

        # Crop the board to remove padding
        board_obs = board_obs[
            0 : -self.env.unwrapped.padding,
            self.env.unwrapped.padding : -self.env.unwrapped.padding,
        ]

        features = []
        if self.report_height or self.report_max_height:
            height_vector = self.calc_height(board_obs)
            if self.report_height: features += list(height_vector)
            if self.report_max_height: features.append(self.aggregate_height(board_obs))

        # Make sure entire board is not all 1s
        if np.any(board_obs == 0): features.append(self.calc_lines(board_obs))
        else: features.append(0)
        if self.report_holes: features.append(self.holes(board_obs))
        if self.report_bumpiness: features.append(self.calc_bumpiness(board_obs))
        features = np.array(features, dtype=np.uint8)
        return features


# Make Grouped Actions Wrapper
def makeGrouped():
    env = makeStandard()
    return GroupedActionsObservationsCopy(env=env, observation_wrappers=[FeatureVectorObservationCopy(env)])
