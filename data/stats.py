import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
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

    os.makedirs("stats", exist_ok=True)
    plt.savefig("stats/BC_rewards.png", dpi=300)

    def gaussian(x, a, b, c):
        return a * np.exp(-b * (x - c)**2)

    # Plot total rewards distribution
    plt.clf()
    hist, edges = np.histogram(total_rewards, bins=100)
    plt.plot(edges[:-1], hist)
    plt.axvline(np.mean(total_rewards), label="Avg", color="black")
    plt.axvline(np.mean(total_rewards) + np.std(total_rewards), label="+1 std. dev.", color="grey")
    plt.axvline(np.mean(total_rewards) - np.std(total_rewards), label="-1 std. dev.", color="grey")

    # Create x, y plot data from histogram
    x_hist = np.zeros(2 * len(edges) - 2)
    y_hist = np.zeros(2 * len(edges) - 2)
    for i in range(len(edges) - 1):
        x_hist[2*i] = edges[i]
        x_hist[2*i + 1] = edges[i + 1]
        y_hist[2*i] = hist[i]
        y_hist[2*i + 1] = hist[i]

    # Fit a gaussian to the histogram
    popt, _ = curve_fit(gaussian, x_hist, y_hist, p0=[1, 1, 3001])
    # print how well the fitting went
    print(f"Standard deviation: {popt[1]}")
    print(f"Mean: {popt[2]}")
    print(f"L2 error: {np.sum((gaussian(x_hist, *popt) - y_hist)**2)}")
    plt.plot(x_hist, gaussian(x_hist, *popt), label="Gaussian fit", color="red")

    plt.xlabel("Total Reward")
    plt.ylabel("Frequency")

    os.makedirs("stats", exist_ok=True)
    plt.savefig("stats/BC_rewards_dist.png", dpi=300)


def pearson_correlation():
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

    total_rewards = np.array(total_rewards)
    A = total_rewards
    B = np.arange(len(total_rewards))
    # print(f"Pearson correlation: {np.corrcoef(total_rewards, total_rewards)[0, 1]}")

    print(f"Pearson correlation: {np.corrcoef(A, B)[0, 1]}")


# find_demonstration_max_total_reward()
plot_demonstrations()
# pearson_correlation()
