import numpy as np


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


# may change the content and transformation in mask and board
def convert_data_state(data: np.ndarray) -> np.ndarray:
    mask   = convert_mask(data['active_tetromino_mask'])
    board  = convert_board(data['board'])
    holder = convert_holder(data['holder'])
    queue  = convert_queue(data['queue'])

    flat1 = (mask + board).flatten()
    flat2 = holder.flatten()
    flat3 = queue.flatten()

    total = np.concatenate([flat1, flat2, flat3])
    return total / np.linalg.norm(total)
