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
val_freq = 100


# Getting training data
Sdata, Adata, Rdata = get_BC_data()
split = int(0.8 * len(Sdata))

Strain, Sval = torch.tensor(Sdata[:split], dtype=torch.float), torch.tensor(Sdata[split:], dtype=torch.float)
Atrain, Aval = torch.tensor(Adata[:split], dtype=torch.float), torch.tensor(Adata[split:], dtype=torch.float)
Rtrain, Rval = torch.tensor(Rdata[:split], dtype=torch.long), torch.tensor(Rdata[split:], dtype=torch.long)


# Convert data to GPU is not on MAC
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Strain, Sval = Strain.to(device), Sval.to(device)
Atrain, Aval = Atrain.to(device), Aval.to(device)
Rtrain, Rval = Rtrain.to(device), Rval.to(device)


# Setting up agent
agent = Agent(STATE_DIM, HIDDEN_DIM, ACTION_DIM, device)
if agent_path: agent.load_path(agent_path)


# Training agent (from HW2)
def train(agent, Strain, Atrain, Sval, Aval, save_path, num_epochs=10, val_freq=100):
    # Setting up loss, optimizer, and data
    training_losses, validation_losses, validation_accs = [], [], []
    best_loss = float('inf')
    best_model = None

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)

    # Training loop across epochs
    for i in range(num_epochs // val_freq):
        for _ in tqdm(range(val_freq), bar_format='{l_bar}{bar:100}{r_bar}{bar:-100b}', leave=False):
            optimizer.zero_grad()
            epoch_loss = loss_fn(agent(Strain), Atrain)

            epoch_loss.backward()
            optimizer.step()

        agent.eval()
        preds = agent(Sval)
        validation_loss = loss_fn(preds, Aval)
        validation_acc = torch.sum(torch.argmax(preds, dim=1) == torch.argmax(Aval, dim=1)) / len(Aval)

        training_losses.append(epoch_loss.item())
        validation_losses.append(validation_loss.item())
        validation_accs.append(validation_acc.item())
        agent.train()

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_model = agent.state_dict()

        ind = str((i+1) * val_freq).rjust(len(str(num_epochs)))
        epoch_loss = str(epoch_loss.item()).ljust(20)
        validation_loss = str(validation_loss.item()).ljust(20)
        validation_acc = str(validation_acc.item()).ljust(20)
        print(f"Epoch {ind} / {num_epochs} - Training Loss: {epoch_loss} - Validation Loss: {validation_loss} - Validation Accuracy: {validation_acc}")

    if save_path:
        agent.load_state(best_model)
        agent.save(save_path)

    return agent, training_losses, validation_losses, validation_accs

agent, training_losses, validation_losses, validation_accs = train(agent, Strain, Atrain, Sval, Aval, save_path, num_epochs=epochs, val_freq=val_freq)


# Graphing validation loss
plt.plot(np.arange(0, epochs, val_freq), training_losses, label="Training Loss", color="red")
plt.plot(np.arange(0, epochs, val_freq), validation_losses, label="Validation Loss", color="green")
plt.plot(np.arange(0, epochs, val_freq), validation_accs, label="Validation Accuracy", color="blue")

plt.title("Training Over Epochs")
plt.xlabel("Epoch")
plt.legend()

stat_path = f"stats/{save_path.split('/')[-1][:-4]}_loss.png"
os.makedirs(os.path.dirname(stat_path), exist_ok=True)
plt.savefig(stat_path)
