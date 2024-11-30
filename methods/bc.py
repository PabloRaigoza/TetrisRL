from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
import torch
import time

from utils.agent import *
from utils.convert import get_BC_data


# Getting command line arguments
parser = argparse.ArgumentParser(description="Train a BC agent")
parser.add_argument("--agent", type=str, default=None, help="Agent path to load")
parser.add_argument("--model", type=str, default="AgentM2", help="Agent version to use")
parser.add_argument("--save", type=str, default=f"agents/BC{int(time.time())}.dat", help="Agent path to save")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
parser.add_argument("--val_freq", type=int, default=100, help="Frequency to validate")

args = parser.parse_args()
agent_path = args.agent
model_class = globals()[args.model]
save_path = args.save
epochs = args.epochs
val_freq = args.val_freq


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
agent = model_class(device)
if agent_path: agent.load_path(agent_path)


# Training agent (from HW2)
def train(agent, Strain, Atrain, Sval, Aval, save_path, num_epochs=10, val_freq=100):
    # Setting up loss, optimizer, and data
    training_losses, validation_losses, validation_accs = [], [], []
    best_loss = float('inf')
    best_model = None

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(agent.parameters(), lr=0.005)


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

    # Saving model and returning
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
