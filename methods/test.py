import numpy as np
import argparse
import cv2

from utils.agent import *
from utils.convert import convert_unwrapped_state, convert_wrapped_state
from utils.environment import makeStandard, makeGrouped, MAX_STEPS


# Getting command line arguments
parser = argparse.ArgumentParser(description="Test an agent")
parser.add_argument("--agent", type=str, default=None, help="Agent path to load")
parser.add_argument("--model", type=str, default="AgentM4", help="Agent version to use")
parser.add_argument("--grouped", type=bool, default=False, help="Whether to use wrapped environment")
parser.add_argument("--attempts", type=int, default=1, help="Number of attempts to test")

device = torch.device("cpu")
args = parser.parse_args()
model_class = globals()[args.model]


# Setup agent and environment
agent = model_class(device)
if args.agent: agent.load_path(args.agent)
env = makeGrouped() if args.grouped else makeStandard()
convert_state = convert_wrapped_state if args.grouped else convert_unwrapped_state
total_reward = 0


# Iterate over the attempts
for i in range(args.attempts):
    seed = np.random.randint(0, 1000000)
    observation = env.reset(seed=seed)

    attempt_reward = 0
    terminated = False
    steps = 0

    # Collect data loop
    while not terminated and steps < MAX_STEPS:
        # Get current state and action
        env.render()
        obs_vector = convert_state(observation[0]) if isinstance(observation, tuple) else convert_state(observation)
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
    ind = str(i+1).rjust(len(str(args.attempts)))
    seed = str(seed).rjust(6)
    reward = str(attempt_reward).rjust(5)
    print(f"Attempt: {ind} / {args.attempts}, Seed: {seed}, Reward: {reward}")

    cv2.waitKey(10)

    # Close the environment
    env.close()
    cv2.destroyAllWindows()


# Print the average reward
print(f"Average reward: {total_reward / args.attempts}")
