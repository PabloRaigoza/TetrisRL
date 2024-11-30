from torch import nn
import numpy as np
import torch
import os


# Constants - (Mask + Board), Holder, Queue
STATE_DIM  = 280
ACTION_DIM = 8


# Base Agent Model
class BaseAgent(nn.Module):
    def get_action(self, state: np.ndarray):
        output = self.forward(state)
        dist = torch.distributions.Categorical(logits=output)
        return dist.sample()


    def load_state(self, state: np.ndarray):
        self.load_state_dict(state)


    def load_path(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))


    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)


# Agent Model 1
class AgentM1(BaseAgent):
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
        return self.fc3(h2) # Return logits


# Agent Model 2
class AgentM2(BaseAgent):
    HIDDEN_DIM = 400

    def __init__(self, device: any, state_dim: int = STATE_DIM,
                 hidden_dim: int = HIDDEN_DIM, action_dim: int = ACTION_DIM):
        super(AgentM2, self).__init__()
        self.state_dim  = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.device     = device

        self.fc1  = nn.Linear(state_dim, hidden_dim).to(device)
        self.fc2  = nn.Linear(hidden_dim, hidden_dim//2).to(device)
        self.fc3  = nn.Linear(hidden_dim//2, hidden_dim//4).to(device)
        self.fc4  = nn.Linear(hidden_dim//4, action_dim).to(device)

        self.leaky = nn.LeakyReLU().to(device)
        self.drop = nn.Dropout().to(device)


    def forward(self, x: np.ndarray):
        x = x.view(-1, self.state_dim)
        h1 = self.drop(self.leaky(self.fc1(x)))
        h2 = self.drop(self.leaky(self.fc2(h1)))
        h3 = self.drop(self.leaky(self.fc3(h2)))
        return self.fc4(h3) # Return logits
