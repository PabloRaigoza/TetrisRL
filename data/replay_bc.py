import numpy as np
import cv2
import os

from utils.environment import makeBC, MAX_STEPS

# Read data directory
LOAD_RAND = False
if LOAD_RAND: file_name = np.random.choice(os.listdir("data/BC"))
else: file_name = "BC_data_106_627639_1000.npy" # Ran using utils/stats.py to find the best replay

# Load data
data = np.load(f"data/BC/{file_name}", allow_pickle=True)

# Create an instance of Tetris
env = makeBC()
observation = env.reset(seed=int(file_name.split("_")[3]))

# Initialize variables
terminated = False
steps = 0

# Print the file name
print(f"Replaying {file_name} with total reward {sum([d['reward'] for d in data])}")

# Collect data loop
while not terminated and steps < MAX_STEPS:
    # Render the current state of the game
    env.render()

    # Pick an action from user input mapped to the keyboard
    action = data[steps]['action']
    obs, reward, terminated, truncated, info = env.step(action)

    # Print the reward
    print(f"Step: {steps}/{MAX_STEPS} Reward: {str(reward).zfill(4)}", end="\r")

    # Update the observation
    observation = obs
    steps += 1

    # Wait for a key press
    cv2.waitKey(1)


# Close the environment
env.close()
cv2.destroyAllWindows()
