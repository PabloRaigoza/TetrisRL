import numpy as np
import argparse
import cv2

from utils.agent import *
from utils.convert import convert_data_state
from utils.environment import makeBC, MAX_STEPS


# Getting command line arguments
parser = argparse.ArgumentParser(description="Test an agent")
parser.add_argument("--agent", type=str, default=None, help="Agent path to load")
parser.add_argument("--model", type=str, default="AgentM2", help="Agent version to use")
parser.add_argument("--attempts", type=int, default=1, help="Number of attempts to test")

device = torch.device("cpu")
args = parser.parse_args()
agent_path = args.agent
model_class = globals()[args.model]
attempts = args.attempts


# Setup agent and environment
agent = model_class(device)
if agent_path: agent.load_path(agent_path)
env = makeBC()
total_reward = 0


# Iterate over the attempts
for i in range(attempts):
    seed = np.random.randint(0, 1000000)
    observation = env.reset(seed=seed)

    attempt_reward = 0
    terminated = False
    steps = 0

    # Collect data loop
    while not terminated and steps < MAX_STEPS:
        # Get current state and action
        env.render()
        obs_vector = convert_data_state(observation[0]) \
            if isinstance(observation, tuple) \
            else convert_data_state(observation)
        action = agent.get_action(torch.tensor(obs_vector, dtype=torch.float32, device=device))

        # Stepping through the environment
        obs, reward, terminated, truncated, info = env.step(action.item())
        attempt_reward += reward
        observation = obs
        steps += 1

        # Delay for visualization
        cv2.waitKey(1)

    # Print the reward
    total_reward += attempt_reward
    ind = str(i+1).rjust(len(str(attempts)))
    seed = str(seed).rjust(6)
    reward = str(attempt_reward).rjust(5)
    print(f"Attempt: {ind} / {attempts}, Seed: {seed}, Reward: {reward}")

    # Close the environment
    env.close()
    cv2.destroyAllWindows()


# Print the average reward
print(f"Average reward: {total_reward / attempts}")
