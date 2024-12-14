import numpy as np
import argparse
import cv2

from utils.agent import *
from utils.convert import convert_unwrapped_state, convert_wrapped_state
from utils.environment import makeStandard, makeGrouped, MAX_STEPS


# # Getting command line arguments
# parser = argparse.ArgumentParser(description="Test an agent")
# parser.add_argument("--agent", type=str, default=None, help="Agent path to load")
# parser.add_argument("--model", type=str, default="AgentM4", help="Agent version to use")
# parser.add_argument("--grouped", type=bool, default=False, help="Whether to use wrapped environment")
# parser.add_argument("--attempts", type=int, default=1, help="Number of attempts to test")

def testModel(agentstr:str, model:str, grouped:bool, attempts:int):
    device = torch.device("cpu")
    # args = parser.parse_args()
    # print(agent, model, grouped, attempts)
    model_class = globals()[model]
    # print(agent, model, grouped, attempts)



    # Setup agent and environment
    agent = model_class(device)
    # print("did i get here?")
    if agentstr: agent.load_path(agentstr)
    env = makeGrouped() if grouped else makeStandard()
    convert_state = convert_wrapped_state if grouped else convert_unwrapped_state
    total_reward = 0

    rewards = []
    # Iterate over the attempts
    for i in range(attempts):
        seed = np.random.randint(0, 1000000)
        observation = env.reset(seed=seed)

        attempt_reward = 0
        terminated = False
        steps = 0

        # Collect data loop
        while not terminated:
            # Get current state and action
            # env.render()
            obs_vector = convert_state(observation[0]) if isinstance(observation, tuple) else convert_state(observation)
            action = agent.get_action(torch.tensor(obs_vector, dtype=torch.float32, device=device))

            # Stepping through the environment
            obs, reward, terminated, truncated, info = env.step(action.item())
            attempt_reward += reward
            observation = obs, info
            steps += 1

            # Delay for visualization
            cv2.waitKey(1)

        # Print the reward
        total_reward += attempt_reward
        rewards.append(attempt_reward)
        ind = str(i+1).rjust(len(str(attempts)))
        seed = str(seed).rjust(6)
        reward = str(attempt_reward).rjust(5)
        print(f"Attempt: {ind} / {attempts}, Seed: {seed}, Reward: {reward}")

        cv2.waitKey(10)

        # Close the environment
        env.close()
        cv2.destroyAllWindows()


    # Print the average reward
    # print(f"Average reward: {total_reward / attempts}")
    mean, std = np.mean(rewards), np.std(rewards)
    print(f"Mean: {mean}, Std: {std}")
    return mean, std

attempts = 100
doTest = False
M1BC = ["agents/M1_BC20000.dat", "AgentM1", False, attempts, "M1BC"]
M2BC = ["agents/M2_BC10000.dat", "AgentM2", False, attempts, "M2BC"]
M3BC = ["agents/M3_BC50000.dat", "AgentM3", True, attempts, "M3BC"]
M4BC = ["agents/M4_BC25000.dat", "AgentM4", True, attempts, "M4BC"]
M4DA_RANDOM = ["agents/M4_1DA25000.dat", "AgentM4", True, attempts, "RAND_DAGGER"]
M4DA_BC = ["agents/M4_2DA25000.dat", "AgentM4", True, attempts, "BC_DAGGER"]
M4DA_DA = ["agents/M4_4DA10000.dat", "AgentM4", True, attempts, "BC_DATA_DAGGER"]
RAND_REINFORCE = ["agents/M4_6R100.dat", "AgentM4", True, attempts, "RAND_REINFORCE"]
BC_REINFORCE = ["agents/M4_3RBC100.dat", "AgentM4", True, attempts, "BC_REINFORCE"]
DA_REINFORCE = ["agents/M4_2RDA100.dat", "AgentM4", True, attempts, "DA_REINFORCE"]
RAND_AVG_REINFORCE = ["agents/M4_2AR100.dat", "AgentM4", True, attempts, "RAND_AVG_REINFORCE"]
BC_AVG_REINFORCE = ["agents/BC_AVG_REINFORCE_0.2.dat", "AgentM4", True, attempts, "BC_AVG_REINFORCE"]
DA_AVG_REINFORCE = ["agents/M4_with0.2.dat", "AgentM4", True, attempts, "DA_AVG_REINFORCE"]

if doTest: toTest = [M1BC, M2BC, M3BC, M4BC, M4DA_RANDOM, M4DA_BC, M4DA_DA, RAND_REINFORCE, BC_REINFORCE, DA_REINFORCE, RAND_AVG_REINFORCE, DA_AVG_REINFORCE, BC_AVG_REINFORCE]
else: toTest = []

data = {}
for test in toTest:
    print("Testing", test[0])
    data[test[4]] = testModel(test[0], test[1], test[2], test[3])

    # with open("methods/data1.npy", "wb") as f: np.save(f, data)


import matplotlib.pyplot as plt
data = np.load("methods/donotdelete_data.npy", allow_pickle=True).item()
data1 = np.load("methods/donotdelete_data1.npy", allow_pickle=True).item()
data2 = np.load("methods/donotdelete_data2.npy", allow_pickle=True).item()
# print(data)

# order = ["M1BC", "M2BC", "M3BC", "M4BC", "M4DA_RANDOM", "M4DA_BC", "M4DA_DA", "RAND_REINFORCE", "BC_REINFORCE", "DA_REINFORCE", "RAND_AVG_REINFORCE", "BC_AVG_REINFORCE", "DA_AVG_REINFORCE"]
# order = []
order = [M1BC, M2BC, M3BC, M4BC, M4DA_RANDOM, M4DA_BC, M4DA_DA, RAND_REINFORCE, BC_REINFORCE, DA_REINFORCE, RAND_AVG_REINFORCE, BC_AVG_REINFORCE, DA_AVG_REINFORCE]
means = []
stds = []
labels = []
# for i in range(len(order)):
for i in range(len(order)):
    # key = order[i]
    key = order[i][4]
    # means.append(data[key][0])
    # stds.append(data[key][1])
    # means.append(np.mean([data[key][0], data1[key][0], data2[key][0]]))
    # stds.append(np.std([data[key][0], data1[key][0], data2[key][0]]))
    # labels.append(key)

    # print(data[key][0], data1[key][0], data2[key][0]) 
    print(key)
    print(data)
    means.append((data[key][0] + data1[key][0] + data2[key][0]) / 3)
    std1 = data[key][1]
    std2 = data1[key][1]
    std3 = data2[key][1]
    # stds.append(np.sqrt((std1**2 + std2**2 + std3**2) / 3))
    stds.append(np.mean([std1, std2, std3]))
    labels.append(key)

fig, ax = plt.subplots(figsize=(15, 6))  # Increase width for a wider chart
x_positions = np.arange(len(labels))

ax.bar(x_positions, means, yerr=stds, capsize=5, color='skyblue', alpha=0.9)
ax.set_xticks(x_positions)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel("Mean Reward")
ax.set_title("Mean Reward with Standard Deviation for Different Models", fontsize=16)
ax.set_xlabel("Models")
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
# set dpi to 300 for better quality

plt.savefig("methods/mean_rewards_with_std_wide1.png", dpi=300)
# plt.show()