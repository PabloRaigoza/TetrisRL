import numpy as np
import cv2
import os

from utils.environment import makeGroupedActionsWrapper, MAX_STEPS
from data.ai import GameMaster
from data.ai import Piece


# Create data directory
data_dir = "data/BC"
os.makedirs(data_dir, exist_ok=True)


# Create data file name
num = len(os.listdir(data_dir))
seed = np.random.randint(0, 1000000)

print(seed)
seed=606239
file_name = f"BC_data_{str(num).zfill(3)}_{str(seed).zfill(6)}_{MAX_STEPS}.npy"


# Create an instance of Tetris
# env = makeBC()
env = makeGroupedActionsWrapper()
observation = env.reset(seed=seed)


# Initialize data array of MAX_STEPS
data = np.array([{} for _ in range(MAX_STEPS)])
terminated = False
steps = 0

ai = GameMaster()

# Collect data loop
# while not terminated and steps < MAX_STEPS:
#     print(seed)
#     # Render the current state of the game
#     env.render()
#     # print(observation[1]['active_tetromino'])
#     act_tet_mat, x, y = observation[1]['active_tetromino']
#     print(act_tet_mat)
#     act_tet = Piece(act_tet_mat)
#     act_tet.row, act_tet.column = y, x

#     original_board = observation[1]['original_board'][:20, 4:14]
#     # mask out the active tetromino
#     mask = np.zeros_like(original_board)
#     # mask[act_tet.row:act_tet.row + act_tet.cells.shape[0], act_tet.column:act_tet.column + act_tet.cells.shape[1]] = act_tet.cells
#     mask[act_tet.column:act_tet.column + act_tet.cells.shape[1], act_tet.row:act_tet.row + act_tet.cells.shape[0]] = act_tet.cells
#     masked_board = np.where(mask, 0, original_board)
#     print(masked_board)

#     # Get the valid actions
#     best_move = ai.start_turn(act_tet, masked_board)
#     print(ai.place_piece(best_move["piece"]))
#     rows, cols = np.where(act_tet_mat != 0)
#     print(best_move["piece"].cells)
#     print(rows, cols)
#     min_col = min(cols)
#     x = best_move["piece"].column + min_col
#     rot = 0
#     for move in best_move["moves"]:
#         if move == "T": rot=(rot+1)%4
#     # # if rot == 4: rot = 0
#     # # # rot = 4 - rot
#     # if rot == 1: x+=1; rot=3
#     # if rot == 3: 
#     #     print("rot 3")
#     #     if x == -1: x+=1
#     #     elif x != 0:
#     #         print("changing from 3 to 1")
#     #         rot = 1
#     #         x-=2
#     # # if rot == 1: x-=1
#     # # elif rot == 3: x+=1

#     # # # def decode_action(self, action):
#     # #     """Converts the action id to the x-position and number of rotations `r`.

#     # #     Args:
#     # #         action: The action id to convert.

#     # #     Returns:
#     # #         The x-position and number of rotations.
#     # #     """ 
#     # #     return action // 4, action % 4
#     # print(x)  
#     # if x < 0:
#     #     print("potential error")
#     #     x = 0
#     # if x > 9:
#     #     print("potential error")
#     #     x = 9
#     # x+=1
#     print(x, rot)
#     # convert x, rot to single value action (0-39)
#     action = x * 4 + rot
#     print(action)
#     action = int(input("Enter action: "))
#     valid_actions = np.where(observation[1]['action_mask'] == 1)[0]
#     random_action = np.random.choice(valid_actions)
#     # obs, reward, terminated, truncated, info = env.step(random_action)
#     obs, reward, terminated, truncated, info = env.step(action)
#     # env.unwrapped

#     # Save the observation
#     data[steps] = {
#         'state': observation,
#         'action': random_action,
#         'reward': reward,
#         'terminated': terminated,
#         'truncated': truncated,
#         'info': info
#     }


#     # Update the observation
#     print(f"Step {steps + 1}/{MAX_STEPS} - Reward: {reward}", end="\r")
#     observation = obs, info
#     steps += 1
#     cv2.waitKey(100000)

i=0
# seed = 728924 # I piece seed

# if True:
while True:
    observation = env.reset(seed=606239)
    env.render()
    # for t in range(len(observation[0])):
    #     print(observation[0][t])
    # action = int(input("Enter action: "))


    # act_tet_mat, x, y = observation[1]['active_tetromino']

    action = i
    print(i)
    obs, reward, terminated, truncated, info = env.step(action)

    env.render()
    cv2.waitKey(100000)
    i+=1

# Close the environment
print(f"Data saved to {os.path.join(data_dir, file_name)}")
env.close()
cv2.destroyAllWindows()
