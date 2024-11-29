from torch import nn
import torch
import os


# Constants
STATE_DIM  = 200 + 16 + 64   # active_tetromino_mask + board, holder, queue
HIDDEN_DIM = 140
ACTION_DIM = 8


# Agent class
class Agent(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super(Agent, self).__init__()
        self.state_dim  = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.fc1  = nn.Linear(state_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3  = nn.Linear(hidden_dim//2, action_dim)
        self.drop = nn.Dropout(p=0.2)


    def forward(self, x):
        x = x.view(-1, self.state_dim)
        h1 = self.drop(torch.relu(self.fc1(x)))
        h2 = self.drop(torch.relu(self.fc2(h1)))
        return torch.sigmoid(self.fc3(h2))


    def load_state(self, state):
        self.load_state_dict(state)


    def load_path(self, path):
        self.load_state_dict(torch.load(path))


    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
