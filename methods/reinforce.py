import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
import torch
import time

from utils.agent import *
from utils.convert import convert_data_state
from utils.environment import makeBC, MAX_STEPS


# Getting command line arguments
parser = argparse.ArgumentParser(description="Train a BC agent")
parser.add_argument("--agent", type=str, default=None, help="Agent path to load")
parser.add_argument("--model", type=str, default="AgentM2", help="Agent version to use")
parser.add_argument("--save", type=str, default=f"agents/REINFORCE{int(time.time())}.dat", help="Agent path to save")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
parser.add_argument("--val_freq", type=int, default=100, help="Frequency to validate")

args = parser.parse_args()
agent_path = args.agent
model_class = globals()[args.model]
save_path = args.save
epochs = args.epochs
val_freq = args.val_freq


# Setting up agent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent = model_class(device)
if agent_path: agent.load_path(agent_path)


# Training agent (from HW4)
class PolicyGradient:
    def __init__(self, env, policy, device, val_freq=100):
        self.env = env
        self.policy = policy
        self.device = device
        self.val_freq = val_freq


    def compute_loss(self, episode, gamma):
        sgen, agen, rgen = zip(*episode)
        ep_states = torch.tensor(sgen, dtype=torch.float32).to(self.device)
        ep_actions = torch.tensor(agen, dtype=torch.int64).to(self.device)
        ep_rewards = torch.tensor(rgen, dtype=torch.float32).to(self.device)
        length = len(episode)

        gammas = (gamma ** torch.arange(length)).to(self.device)
        discounted = ep_rewards * gammas
        rewards_togo = discounted.sum() - torch.cumsum(discounted, dim=0) + discounted

        action_logits = self.policy(ep_states)
        action_dists = torch.softmax(action_logits, dim=1)
        action_probs = action_dists[torch.arange(length), ep_actions].to(self.device)
        loss = - torch.sum(torch.log(action_probs) * rewards_togo.to(self.device))
        return loss


    def update_policy(self, episodes, optimizer, gamma):
        optimizer.zero_grad()
        losses = torch.stack([self.compute_loss(episode, gamma) for episode in episodes])
        avg_loss = torch.mean(losses.to(self.device))
        avg_loss.backward(); optimizer.step()
        return avg_loss.item()


    def run_episode(self):
        seed = np.random.randint(0, 1000000)
        state = self.env.reset(seed=seed)
        episode = []
        terminated = False

        while not terminated and len(episode) < MAX_STEPS:
            obs_vector = convert_data_state(state[0]) \
                if isinstance(state, tuple) \
                else convert_data_state(state)
            action = self.policy.get_action(torch.tensor(obs_vector, dtype=torch.float32, device=self.device))

            next_state, reward, terminated, _, _ = self.env.step(action.item())
            episode.append((obs_vector, action, reward))
            state = next_state
        return episode


    def train(self, num_iterations, batch_size, gamma, lr):
        """Train the policy network using the REINFORCE algorithm

        Args:
            num_iterations (int): Number of iterations to train the policy network
            batch_size (int): Number of episodes per batch
            gamma (float): Discount factor
            lr (float): Learning rate
        """
        self.policy.train()
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        avg_losses, avg_rewards = [], []
        best_loss = float('inf')
        best_model = None

        for i in range(num_iterations):
            episodes = [self.run_episode() for _ in tqdm(range(batch_size), bar_format='{l_bar}{bar:100}{r_bar}{bar:-100b}', leave=False)]
            avg_loss = self.update_policy(episodes, optimizer, gamma)

            if (i % self.val_freq == 0):
                avg_losses.append(avg_loss)
                avg_reward = self.evaluate(batch_size, gamma)
                avg_rewards.append(avg_reward)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model = self.policy.state_dict()

                ind = str(i).rjust(len(str(num_iterations)))
                avg_loss = str(avg_loss).ljust(20)
                avg_reward = str(avg_reward).ljust(20)
                print(f"Epoch {ind} / {num_iterations} - Training Loss: {avg_loss} - Validation Reward {avg_reward}")

        if save_path:
            agent.load_state(best_model)
            agent.save(save_path)

        return avg_losses, avg_rewards


    def evaluate(self, num_episodes, gamma):
        self.policy.eval()
        total_reward = 0

        for _ in range(num_episodes):
            episode = self.run_episode()
            _, _, rgen = zip(*episode)

            rewards = torch.tensor(rgen, dtype=torch.float32).to(self.device)
            gammas = (gamma ** torch.arange(len(rewards))).to(self.device)
            total_reward += (rewards * gammas).sum()

        return (total_reward / num_episodes).item()


reinforce = PolicyGradient(makeBC(), agent, device, val_freq)
avg_losses, avg_rewards = reinforce.train(epochs, batch_size=10, gamma=1, lr=0.005)


# Graphing losses and rewards
plt.plot(np.arange(0, epochs, val_freq), avg_losses, label="Training Loss", color="red")
plt.plot(np.arange(0, epochs, val_freq), avg_rewards, label="Validation Reward", color="blue")

plt.title("Training Over Epochs")
plt.xlabel("Epoch")
plt.legend()

stat_path = f"stats/{save_path.split('/')[-1][:-4]}_loss.png"
os.makedirs(os.path.dirname(stat_path), exist_ok=True)
plt.savefig(stat_path)
