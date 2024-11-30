from torch import nn
import numpy as np
import torch
import os


# Constants
STATE_DIM  = 280
ACTION_DIM = 8


# Agent Model 1
class AgentM1(nn.Module):
    # Constants - (Mask + Board), Holder, Queue
    HIDDEN_DIM = 140

    def __init__(self, device: any, state_dim: int = STATE_DIM,
                 hidden_dim: int = HIDDEN_DIM, action_dim: int = ACTION_DIM):
        super(AgentM1, self).__init__()
        self.state_dim  = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.device     = device

        self.fc1  = nn.Linear(state_dim, hidden_dim).to(device)
        self.fc2  = nn.Linear(hidden_dim, hidden_dim//2).to(device)
        self.fc3  = nn.Linear(hidden_dim//2, action_dim).to(device)
        self.drop = nn.Dropout(p=0.2).to(device)


    def forward(self, x: np.ndarray):
        x = x.view(-1, self.state_dim)
        h1 = self.drop(torch.relu(self.fc1(x)))
        h2 = self.drop(torch.relu(self.fc2(h1)))
        return torch.sigmoid(self.fc3(h2))


    def get_action(self, state: np.ndarray):
        return torch.argmax(self.forward(state)).item()


    def load_state(self, state: np.ndarray):
        self.load_state_dict(state)


    def load_path(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))


    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)


# Agent Model 2
class AgentM2(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, device: any):
        super(AgentM1, self).__init__()
        self.state_dim  = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.device     = device

        self.fc1  = nn.Linear(state_dim, hidden_dim).to(device)
        self.fc2  = nn.Linear(hidden_dim, hidden_dim//2).to(device)
        self.fc3  = nn.Linear(hidden_dim//2, action_dim).to(device)
        self.drop = nn.Dropout(p=0.2).to(device)
