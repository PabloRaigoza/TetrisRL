import numpy as np
import cv2
import os

from utils.environment import makeBC, MAX_STEPS


# Create data directory
data_dir = "data/BC"
os.makedirs(data_dir, exist_ok=True)


# Create data file name
num = len(os.listdir(data_dir))
seed = np.random.randint(0, 1000000)
file_name = f"BC_data_{str(num).zfill(3)}_{str(seed).zfill(6)}_{MAX_STEPS}.npy"


# Create an instance of Tetris
env = makeBC()
observation = env.reset(seed=seed)


# Initialize data array of MAX_STEPS
data = np.array([{} for _ in range(MAX_STEPS)])
terminated = False
steps = 0


# Collect data loop
while not terminated and steps < MAX_STEPS:
    # Render the current state of the game
    env.render()


    # Pick an action from user input mapped to the keyboard
    action = None
    while action is None:
        key = cv2.waitKey(1)
        if key == ord("a"):   action = env.unwrapped.actions.move_left
        elif key == ord("d"): action = env.unwrapped.actions.move_right
        elif key == ord("s"): action = env.unwrapped.actions.move_down
        elif key == ord(";"): action = env.unwrapped.actions.rotate_counterclockwise
        elif key == ord("'"): action = env.unwrapped.actions.rotate_clockwise
        elif key == ord(" "): action = env.unwrapped.actions.hard_drop
        elif key == ord("c"): action = env.unwrapped.actions.swap
        elif key == ord("t"): action = env.unwrapped.actions.no_op

    obs, reward, terminated, truncated, info = env.step(action)


    # Save the observation
    data[steps] = {
        'state': observation,
        'action': action,
        'reward': reward,
        'terminated': terminated,
        'truncated': truncated,
        'info': info
    }

    with open(os.path.join(data_dir, file_name), "wb") as f:
        np.save(f, data)


    # Update the observation
    print(f"Step {steps + 1}/{MAX_STEPS} - Reward: {reward}", end="\r")
    observation = obs
    steps += 1


# Close the environment
print(f"Data saved to {os.path.join(data_dir, file_name)}")
env.close()
cv2.destroyAllWindows()
