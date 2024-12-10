from multiprocessing import Pool
import numpy as np
import argparse
import cv2
import os

from utils.environment import makeGrouped, MAX_STEPS
from utils.agent import ExpertAgent


# Getting command line arguments
parser = argparse.ArgumentParser(description="Test an agent")
parser.add_argument("--attempts", type=int, default=1, help="Number of attempts to test")
parser.add_argument("--threads", type=int, default=1, help="Number of threads to use")
args = parser.parse_args()


# Create data directory
data_dir = "data/BC"
os.makedirs(data_dir, exist_ok=True)

def collect_data(i, base, seed):
    # Create data file name
    file_name = f"BC_data_{str(i+base).zfill(3)}_{str(seed).zfill(6)}_{MAX_STEPS}.npy"


    # Create an instance of Tetris
    env = makeGrouped()
    observation = env.reset(seed=seed)
    expert = ExpertAgent()


    # Initialize data array of MAX_STEPS
    data = np.array([{} for _ in range(MAX_STEPS)])
    terminated = False
    steps = 0


    # Collect data loop
    while not terminated and steps < MAX_STEPS:
        # env.render()
        action = expert.get_action(observation[0], observation[1]["action_mask"])
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
        observation = obs, info
        steps += 1
        cv2.waitKey(1)


    # Close the environment
    print(f"Data saved to {os.path.join(data_dir, file_name)} - Steps: {steps}")
    env.close()
    cv2.destroyAllWindows()


# Collect data with multiprocessing
if __name__ == "__main__":
    with Pool(args.threads) as p:
        start = len(os.listdir(data_dir))
        seeds = np.random.randint(0, 1e7, args.attempts, dtype=np.int32)
        p.starmap(collect_data, [(i, start, int(seed)) for i, seed in enumerate(seeds)])
