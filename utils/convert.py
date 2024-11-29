import numpy as np
import os

from utils.agent import ACTION_DIM

def convert_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask[:-4, 4:-4]
    return mask


def convert_board(board: np.ndarray) -> np.ndarray:
    board = board[:-4, 4:-4]
    board[board != 0] = 1
    return board


def convert_holder(holder: np.ndarray) -> np.ndarray:
    holder[holder != 0] = 1
    return holder


def convert_queue(queue: np.ndarray) -> np.ndarray:
    queue[queue != 0] = 1
    return queue


def convert_data_state(data: dict) -> np.ndarray:
    mask   = convert_mask(data['active_tetromino_mask'])
    board  = convert_board(data['board'])
    holder = convert_holder(data['holder'])
    queue  = convert_queue(data['queue'])

    flat1 = (mask + board).flatten()
    flat2 = holder.flatten()
    flat3 = queue.flatten()

    total = np.concatenate([flat1, flat2, flat3])
    return total / np.linalg.norm(total)


def convert_data_action(data: int) -> np.ndarray:
    action = np.zeros(ACTION_DIM)
    action[data] = 1
    return action


def get_BC_data() -> np.ndarray:
    path = 'data/BC'
    Sdata, Adata, Rdata = [], [], []
    print(f'Loading data from {path}...')

    # iterate through all files in the directory
    for file in os.listdir(path):
        file_data = np.load(f'{path}/{file}', allow_pickle=True)

        for i, data in enumerate(file_data):
            # assert len(data['state']) == 2, f'State length is not 2: {len(data["state"])}, {file}, {i}, \n{data["state"][0]}'
            if isinstance(data['state'], tuple):
                Sdata.append(convert_data_state(data['state'][0]))
                Adata.append(convert_data_action(data['action']))
                Rdata.append(data['reward'])
            else:
                Sdata.append(convert_data_state(data['state']))
                Adata.append(convert_data_action(data['action']))
                Rdata.append(data['reward'])

    perm = np.random.permutation(len(Sdata))
    Sdata = np.array(Sdata)[perm]
    Adata = np.array(Adata)[perm]
    Rdata = np.array(Rdata)[perm]
    return Sdata, Adata, Rdata
