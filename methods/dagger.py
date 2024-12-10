import numpy as np
from tqdm import tqdm
import argparse
import torch
import time
import cv2
import os

from utils.agent import *
from utils.convert import get_wrapped_DA_data, convert_wrapped_state
from utils.environment import makeGrouped, MAX_STEPS


# Getting command line arguments
parser = argparse.ArgumentParser(description="Train a DAgger agent")
parser.add_argument("--agent", type=str, default=None, help="Agent path to load")
parser.add_argument("--model", type=str, default="AgentM4", help="Agent version to use")
parser.add_argument("--save", type=str, default=f"agents/DA{int(time.time())}.dat", help="Agent path to save")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
parser.add_argument("--val_freq", type=int, default=100, help="Frequency to validate")

args = parser.parse_args()
model_class = globals()[args.model]


# Getting training data
Sdata, Edata = get_wrapped_DA_data()
split = int(0.8 * len(Sdata))

Strain, Sval = torch.tensor(Sdata[:split], dtype=torch.float), torch.tensor(Sdata[split:], dtype=torch.float)
Etrain, Eval = torch.tensor(Edata[:split], dtype=torch.long), torch.tensor(Edata[split:], dtype=torch.long)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Strain, Sval = Strain.to(device), Sval.to(device)
Etrain, Eval = Etrain.to(device), Eval.to(device)


# Setting up agent
expert = ExpertAgent(device)
agent = model_class(device)
if args.agent: agent.load_path(args.agent)


# Function for collecting expert data
def collect_expert_data(agent, Strain, Sval, Etrain, Eval):
    # Create data directory
    data_dir = "data/DA"
    os.makedirs(data_dir, exist_ok=True)

    # Create data file name
    num = len(os.listdir(data_dir))
    seed = np.random.randint(0, 1000000)
    file_name = f"DA_data_{str(num).zfill(3)}_{str(seed).zfill(6)}_{MAX_STEPS}.npy"

    # Create an instance of Tetris
    env = makeGrouped()
    observation = env.reset(seed=seed)
    NSdata, NEdata = [], []

    # Initialize data array of MAX_STEPS
    data = np.array([{} for _ in range(MAX_STEPS)])
    terminated = False
    steps = 0


    # Collect data loop
    while not terminated and steps < MAX_STEPS:
        # Render the current state of the game
        # env.render()

        # Get state and action from agent
        obs_vector = convert_wrapped_state(observation[0]) \
                if isinstance(observation, tuple) \
                else convert_wrapped_state(observation)
        action = agent.get_action(torch.tensor(obs_vector, dtype=torch.float32, device=device))

        # Get expert action from user input and update the state
        expert_action = expert.get_action(obs_vector, observation[1]['action_mask'])
        obs, reward, terminated, truncated, info = env.step(action.item())

        # Save the observation and append data
        data[steps] = {
            'state': observation,
            'action': action,
            'expert_action': expert_action,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
            'info': info
        }

        NSdata.append(obs_vector)
        NEdata.append(expert_action)

        with open(os.path.join(data_dir, file_name), "wb") as f:
            np.save(f, data)

        # Update the observation
        observation = obs, info
        steps += 1
        cv2.waitKey(1)


    # Close the environment
    env.close()
    cv2.destroyAllWindows()

    split = int(0.8 * len(NSdata))
    NStrain, NSval = torch.tensor(NSdata[:split], dtype=torch.float), torch.tensor(NSdata[split:], dtype=torch.float)
    NEtrain, NEval = torch.tensor(NEdata[:split], dtype=torch.long), torch.tensor(NEdata[split:], dtype=torch.long)

    Strain, Sval = torch.cat((Strain, NStrain), dim=0), torch.cat((Sval, NSval), dim=0)
    Etrain, Eval = torch.cat((Etrain, NEtrain), dim=0), torch.cat((Eval, NEval), dim=0)
    return Strain, Sval, Etrain, Eval


# Training agent (from HW2)
def train(agent, Strain, Etrain, Sval, Eval, save_path, num_epochs=10, val_freq=100):
    # Setting up loss, optimizer, and data
    training_losses, validation_losses, validation_accs = [], [], []
    best_loss = float('inf')
    best_model = None

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(agent.parameters(), lr=0.005)


    # Training loop across epochs
    for i in range(num_epochs // val_freq):
        print("Collecting 5 expert trajectories...")
        for _ in range(5):
            Strain, Sval, Etrain, Eval = collect_expert_data(agent, Strain, Sval, Etrain, Eval)

        for _ in tqdm(range(val_freq), bar_format='{l_bar}{bar:100}{r_bar}{bar:-100b}', leave=False):
            optimizer.zero_grad()
            epoch_loss = loss_fn(agent(Strain), Etrain)

            epoch_loss.backward()
            optimizer.step()

        agent.eval()
        preds = agent(Sval)
        validation_loss = loss_fn(preds, Eval)
        validation_acc = torch.sum(agent.get_action(Sval) == Eval).float() / len(Eval)

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


agent, training_losses, validation_losses, validation_accs = \
    train(agent, Strain, Etrain, Sval, Eval, args.save, num_epochs=args.epochs, val_freq=args.val_freq)
