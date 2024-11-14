from tetris_gymnasium.envs import Tetris
import gymnasium as gym
import numpy as np
import sys
import cv2
import os

MAX_STEPS = 1000

# Create data directory
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
seed = np.random.randint(0, 100000)
file_name = f"BC_data_{len(os.listdir(data_dir))}_{seed}_{MAX_STEPS}.npy"

# Create an instance of Tetris
env = gym.make("tetris_gymnasium/Tetris", render_mode="human", render_upscale=30)
env.reset(seed=seed)

# Initialize data array of MAX_STEPS
data = np.array([{} for _ in range(MAX_STEPS)])
# Main game loop
terminated = False
steps = 0
while not terminated and steps < MAX_STEPS:
    # Render the current state of the game as text
    env.render()

    # Pick an action from user input mapped to the keyboard
    action = None
    while action is None:
        key = cv2.waitKey(1)
        if key == ord("a"): action = env.unwrapped.actions.move_left
        elif key == ord("d"): action = env.unwrapped.actions.move_right
        elif key == ord("s"): action = env.unwrapped.actions.move_down
        elif key == ord(";"): action = env.unwrapped.actions.rotate_counterclockwise
        elif key == ord("'"): action = env.unwrapped.actions.rotate_clockwise
        elif key == ord(" "): action = env.unwrapped.actions.hard_drop
        elif key == ord("c"): action = env.unwrapped.actions.swap
        elif key == ord("t"): action = env.unwrapped.actions.no_op

    # Perform the action
    observation, reward, terminated, truncated, info = env.step(action)

    # Save the observation
    observation['action'] = action
    observation['reward'] = reward
    observation['terminated'] = terminated
    observation['info'] = info
    observation['truncated'] = truncated

    # Save the observation
    # with open(os.path.join(data_dir, file_name), "ab") as f:
    #     np.save(f, observation)
    data[steps] = observation

    with open(os.path.join(data_dir, file_name), "wb") as f:
        np.save(f, data)

    print(f"Step {steps}")
    steps += 1

# Close the environment
env.close()
cv2.destroyAllWindows()
print(f"Data saved to {os.path.join(data_dir, file_name)}")