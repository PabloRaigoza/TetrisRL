import numpy as np
import os


# Function to convert data dictionary to unwrapped state
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


def convert_unwrapped_state(data: dict) -> np.ndarray:
    mask   = convert_mask(data['active_tetromino_mask'])
    board  = convert_board(data['board'])
    holder = convert_holder(data['holder'])
    queue  = convert_queue(data['queue'])

    flat1 = (mask + board).flatten()
    flat2 = holder.flatten()
    flat3 = queue.flatten()

    total = np.concatenate([flat1, flat2, flat3])
    return total / np.linalg.norm(total)


def get_unwrapped_BC_data() -> np.ndarray:
    path = 'data/oldBC'
    Sdata, Adata = [], []
    print(f'Loading data from {path}...')

    if not os.path.exists(path):
        return np.array(Sdata), np.array(Adata)

    # Iterate through all files in the directory
    for file in os.listdir(path):
        file_data = np.load(f'{path}/{file}', allow_pickle=True)

        for data in file_data:
            if isinstance(data['state'], tuple):
                Sdata.append(convert_unwrapped_state(data['state'][0]))
                Adata.append(data['action'])
            else:
                Sdata.append(convert_unwrapped_state(data['state']))
                Adata.append(data['action'])

    perm = np.random.permutation(len(Sdata))
    Sdata = np.array(Sdata)[perm]
    Adata = np.array(Adata)[perm]
    return Sdata, Adata


# Functions to convert data dictionary to wrapped state
def convert_wrapped_state(data: dict) -> np.ndarray:
    return data.flatten()


def get_wrapped_BC_data(rand: bool = False) -> np.ndarray:
    path = 'data/BC'
    Sdata, Adata = [], []
    print(f'Loading data from {path}...')

    if not os.path.exists(path):
        return np.array(Sdata), np.array(Adata)

    # Iterate through all files in the directory
    for file in os.listdir(path):
        file_data = np.load(f'{path}/{file}', allow_pickle=True)
        not_empty = [data for data in file_data if len(data) > 0]
        if (len(not_empty) < 500): continue

        # Assign 50% probability to load data
        if rand and (np.random.rand() < 0.5): continue

        for data in not_empty:
            if isinstance(data['state'], tuple):
                Sdata.append(convert_wrapped_state(data['state'][0]))
                Adata.append(data['action'])
            else:
                Sdata.append(convert_wrapped_state(data['state']))
                Adata.append(data['action'])

    perm = np.random.permutation(len(Sdata))
    Sdata = np.array(Sdata)[perm]
    Adata = np.array(Adata)[perm]
    return Sdata, Adata
