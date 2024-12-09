import numpy as np
import cv2
import os

# from utils.environment import makeBC, MAX_STEPS
from tetris import Tetris as extTetris
from tetris_gymnasium.mappings.rewards import RewardsMapping
from tetris_gymnasium.envs import *
import gymnasium as gym

MAX_STEPS = 1000


# Create data directory
data_dir = "data/BC"
os.makedirs(data_dir, exist_ok=True)


# Create data file name
num = len(os.listdir(data_dir))
seed = np.random.randint(0, 1000000)
file_name = f"BC_data_{str(num).zfill(3)}_{str(seed).zfill(6)}_{MAX_STEPS}.npy"


# Create an instance of Tetris
# env = makeBC()
from tetris_gymnasium.mappings.rewards import RewardsMapping
rewards_mapping = RewardsMapping()
rewards_mapping.game_over = -100
rewards_mapping.invalid_action = -10
env = gym.make(
        "tetris_gymnasium/Tetris",
        render_mode="human",
        render_upscale=40,
        rewards_mapping=rewards_mapping
    )
observation = env.reset(seed=seed)


# Initialize data array of MAX_STEPS
data = np.array([{} for _ in range(MAX_STEPS)])
terminated = False
steps = 0

import torch
import argparse
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="data")
    parser.add_argument("--output", type=str, default="output.mp4")

    args = parser.parse_args()
    return args

opt = get_args()

if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
else:
    torch.manual_seed(123)
if torch.cuda.is_available():
    model = torch.load("tetris".format(opt.saved_path))
else:
    model = torch.load("tetris".format(opt.saved_path), map_location=lambda storage, loc: storage)

# Collect data loop
exttetris = extTetris()
exttetris.reset()
while not terminated and steps < MAX_STEPS:
    # Render the current state of the game
    env.render()

    # I 2 -> 0xAA0000 5
    # O 3 -> 0x0000AA 1
    # T 4 -> 0xAA5500 2
    # S 5 -> 0x00AA00 3
    # Z 6 -> 0x00AAAA 4
    # J 7 -> 0x5500AA 7
    # L 8 -> 0xAA00AA 6
    # mappings = [0,1,5,1,2,3,4,7,6]
    mappings =   [0,-1,4,0,1,2,3,6,5]
    cur_board = observation[0]['board'][:20, 4:14] if steps == 0 else observation['board'][:20, 4:14]
    cur_active_piece = observation[0]['active_tetromino_mask'][:20,4:14] if steps == 0 else observation['active_tetromino_mask'][:20,4:14]
    
    cur_piece_ind = mappings[np.max(cur_active_piece)]
    
    rows, cols = np.where(cur_active_piece != 0)
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    piece_mask = cur_active_piece[min_row:max_row+1, min_col:max_col+1]
    pos = {"x": 5 - len(piece_mask[0]) // 2, "y": (min_row + max_row) // 2}
    # pos = {"x": min_row, "y": min_col}
    cur_board[cur_board > 1] -= 1
    # self.board = [[0] * self.width for _ in range(self.height)]

    new_board = [[cur_board[i][j] for j in range(10)] for i in range(20)]
    exttetris.restart(
        # list(cur_board),
        new_board,
        cur_piece_ind,
        pos,
    )
    exttetris.render()
    cv2.waitKey(1000000)
    next_steps = exttetris.get_next_states()
    next_actions, next_states = zip(*next_steps.items())
    next_states = torch.stack(next_states)
    if torch.cuda.is_available(): next_states = next_states.cuda()
    x, rotations = next_actions[torch.argmax(model(next_states)[:, 0]).item()]
    
    # Determine next action from the model
    # x is the column to place the piece
    # rotations is the number of rotations to perform
    action = env.unwrapped.actions.no_op
    if rotations > 0:
        action = env.unwrapped.actions.rotate_clockwise
    else:
        if x < pos["x"]:
            action = env.unwrapped.actions.move_left
        elif x > pos["x"]:
            action = env.unwrapped.actions.move_right
        else:
            action = env.unwrapped.actions.hard_drop
    obs, reward, terminated, truncated, info = env.step(action)

    steps += 1
    observation = obs

    env.render()
    cv2.waitKey(1000000)


# Close the environment
print(f"Data saved to {os.path.join(data_dir, file_name)}")
env.close()
cv2.destroyAllWindows()
