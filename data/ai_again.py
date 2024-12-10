import numpy as np
import cv2
import os

from utils.environment import makeGroupedActionsWrapper, MAX_STEPS


env = makeGroupedActionsWrapper()
# observation = env.reset(seed=np.random.randint(0, 1000000))
observation = env.reset(seed=728924)

# Initialize variables
terminated = False
steps = 0


class Grid:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.cells = np.zeros((rows, columns))

    def clone(self):
        _grid = Grid(self.rows, self.columns)
        _grid.cells = np.copy(self.cells)
        return _grid

    def clear_lines(self):
        distance = 0
        for r in range(self.rows - 1, -1, -1):
            if self.is_line(r):
                distance += 1
                self.cells[r] = np.zeros(self.columns)
            elif distance > 0:
                self.cells[r + distance] = self.cells[r]
                self.cells[r] = np.zeros(self.columns)
        return distance

    def is_line(self, row):
        return not np.any(self.cells[row] == 0)

    def is_empty_row(self, row):
        return not np.any(self.cells[row] != 0)

    def exceeded(self):
        return not self.is_empty_row(0) or not self.is_empty_row(1)

    def height(self):
        r = 0
        while r < self.rows and self.is_empty_row(r):
            r += 1
        return self.rows - r
    
    def lines(self):
        return sum([self.is_line(r) for r in range(self.rows)])
    
    def holes(self):
        count = 0

        for c in range(self.columns):
            hole = False
            for r in range(self.rows):
                if self.cells[r][c] != 0:
                    hole = True
                elif self.cells[r][c] == 0 and hole:
                    count += 1

        return count

    def blockades(self):
        count = 0

        for c in range(self.columns):
            hole = False
            for r in range(self.rows - 1, -1, -1):
                if self.cells[r][c] == 0:
                    hole = True
                elif self.cells[r][c] != 0 and hole:
                    count += 1

        return count

    def aggregate_height(self):
        return sum([self.column_height(c) for c in range(self.columns)])

    def bumpiness(self):
        return sum([abs(self.column_height(c) - self.column_height(c + 1)) for c in range(self.columns - 1)])

    def column_height(self, column):
        r = 0
        while r < self.rows and self.cells[r][column] == 0:
            r += 1
        return self.rows - r

    def add_piece(self, piece):
        for r in range(len(piece.cells)):
            for c in range(len(piece.cells[r])):
                _r = piece.row + r
                _c = piece.column + c
                if piece.cells[r][c] != 0 and _r >= 0:
                    self.cells[_r][_c] = piece.cells[r][c]

    def valid(self, piece):
        for r in range(len(piece.cells)):
            for c in range(len(piece.cells[r])):
                _r = piece.row + r
                _c = piece.column + c
                if piece.cells[r][c] != 0:
                    if _r < 0 or _r >= self.rows:
                        return False
                    if _c < 0 or _c >= self.columns:
                        return False
                    if self.cells[_r][_c] != 0:
                        return False
        return True

# Collect data loop
while not terminated and steps < MAX_STEPS:
    # Render the current state of the game
    env.render()

    # Get all possible boards
    # (40, 24, 18)
    boards = observation[0][:, :20, 4:14]
    # print(boards)
    # for i in range(40):
    #     # cv2.imshow(f"Board {i}", boards[i])
    #     print(boards[i])
    # print(observation[1]["action_mask"])
    best_score = -np.inf
    best_board_index = -1
# -heightWeight * _grid.aggregate_height() + linesWeight * _grid.lines() - holesWeight * _grid.holes() - bumpinessWeight * _grid.bumpiness()
    # Evaluate all possible boards
    _grid = Grid(20, 10)
    heightWeight = 0.510066
    linesWeight = 0.760666
    holesWeight = 0.35663
    bumpinessWeight = 0.184483
    # for i in range(40):
    for i, board in enumerate(boards):
        if observation[1]["action_mask"][i] == 0:
            continue
        _grid.cells = board
        # score = -_grid.aggregate_height() + _grid.lines() - _grid.holes() - _grid.bumpiness()
        score = -heightWeight * _grid.aggregate_height() + linesWeight * _grid.lines() - holesWeight * _grid.holes() - bumpinessWeight * _grid.bumpiness()
        if score > best_score:
            best_score = score
            best_board_index = i


    # print(observation[0].shape)

    # Pick an action from user input mapped to the keyboard
    action = np.random.randint(0,7)
    action = best_board_index
    if observation[1]["action_mask"][action] == 0:
        print("Invalid action")
    obs, reward, terminated, truncated, info = env.step(action)

    # Print the reward
    # print(f"Step: {steps}/{MAX_STEPS} Reward: {str(reward).zfill(4)}", end="\r")

    # Update the observation
    observation = obs, info
    steps += 1

    # Wait for a key press
    cv2.waitKey(1)


# Close the environment
env.close()
cv2.destroyAllWindows()
