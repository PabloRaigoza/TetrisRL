from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import time

from utils.agent import *
from utils.convert import get_BC_data


# Getting user arguments
agent_path = input("Path to the agent: ") or None
save_path = input("Path to save the agent: ") or f"agents/BC{int(time.time())}.dat"
epochs = int(input("Number of epochs: ") or 10)
val_freq = 5


# Getting training data
Sdata, Adata, Rdata = get_BC_data()
split = int(0.8 * len(Sdata))

Strain, Sval = torch.tensor(Sdata[:split], dtype=torch.float), torch.tensor(Sdata[split:], dtype=torch.float)
Atrain, Aval = torch.tensor(Adata[:split], dtype=torch.float), torch.tensor(Adata[split:], dtype=torch.float)
Rtrain, Rval = torch.tensor(Rdata[:split], dtype=torch.long), torch.tensor(Rdata[split:], dtype=torch.long)


# Convert data to GPU is not on MAC
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Strain, Sval = Strain.to(device), Sval.to(device)
Atrain, Aval = Atrain.to(device), Aval.to(device)
Rtrain, Rval = Rtrain.to(device), Rval.to(device)


# Setting up agent
agent = Agent(STATE_DIM, HIDDEN_DIM, ACTION_DIM)
if agent_path: agent.load_path(agent_path)


# Training agent (from HW2)
def train(agent, Strain, Atrain, Sval, Aval, save_path, num_epochs=10, val_freq=5):
    # Setting up loss, optimizer, and data
    validation_losses = []
    best_loss = float('inf')
    best_model = None

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)

    dataset = TensorDataset(Strain, Atrain)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


    # Training loop across epochs
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        agent.train()

        for states, actions in dataloader:
            optimizer.zero_grad()
            loss = loss_fn(agent(states), actions)
            epoch_loss += loss

            loss.backward()
            optimizer.step()

        if epoch % val_freq == 0:
            agent.eval()
            validation_loss = loss_fn(agent(Sval), Aval)
            validation_losses.append(validation_loss.item())

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_model = agent.state_dict()

    if save_path:
        agent.load_state(best_model)
        agent.save(save_path)

    return agent, validation_losses
agent, validation_losses = train(agent, Strain, Atrain, Sval, Aval, save_path, num_epochs=epochs, val_freq=val_freq)


# Graphing validation loss
plt.plot(np.arange(0, epochs, val_freq), validation_losses)
plt.title("Validation Loss vs. Epochs")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")

stat_path = f"stats/{save_path.split('/')[-1][:-4]}_loss.png"
os.makedirs(os.path.dirname(stat_path), exist_ok=True)
plt.savefig(stat_path)
