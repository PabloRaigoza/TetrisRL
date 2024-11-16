import matplotlib.pyplot as plt
import numpy as np
import os

def find_demonstration_max_total_reward():
    # Read data directory
    data_dir = "data/BC"

    # Search for replay with higest total reward
    max_reward = -np.inf
    max_file = None
    for file in os.listdir(data_dir):
        data = np.load(os.path.join(data_dir, file), allow_pickle=True)
        total_reward = sum([d['reward'] for d in data])
        if total_reward > max_reward:
            max_reward = total_reward
            max_file = file

    print(f"Replaying {max_file} with total reward {max_reward}")

def plot_demonstrations():
    # Read data directory
    data_dir = "data/BC"

    # Get the total rewards
    total_rewards = []
    for i in range(len(os.listdir(data_dir))):
        file_name = None
        for file in os.listdir(data_dir):
            if file.split("_")[2] == str(i).zfill(3):
                file_name = file
                break
        data = np.load(os.path.join(data_dir, file_name), allow_pickle=True)
        total_reward = sum([d['reward'] for d in data])
        total_rewards.append(total_reward)

    # Plot the total rewards
    plt.plot(total_rewards)
    plt.plot([np.mean(total_rewards)]*len(total_rewards), label="Avg", color="black")
    plt.plot([np.mean(total_rewards) + np.std(total_rewards)]*len(total_rewards), label="+1 std. dev.", color="grey")
    plt.plot([np.mean(total_rewards) - np.std(total_rewards)]*len(total_rewards), label="-1 std. dev.", color="grey")
    plt.xlabel("Demonstration")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig("total_rewards.png", dpi=300)

    # Plot total rewards distribution
    plt.clf()
    hist, edges = np.histogram(total_rewards, bins=100)
    plt.plot(edges[:-1], hist)
    plt.axvline(np.mean(total_rewards), label="Avg", color="black")
    plt.axvline(np.mean(total_rewards) + np.std(total_rewards), label="+1 std. dev.", color="grey")
    plt.axvline(np.mean(total_rewards) - np.std(total_rewards), label="-1 std. dev.", color="grey")
    plt.xlabel("Total Reward")
    plt.ylabel("Frequency")
    plt.savefig("total_rewards_distribution.png", dpi=300)

# find_demonstration_max_total_reward()
plot_demonstrations()
