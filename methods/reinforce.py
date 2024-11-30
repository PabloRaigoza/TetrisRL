import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
import torch
import time

from utils.agent import *

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
