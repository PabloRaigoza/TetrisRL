import numpy as np

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

class Piece:
    def __init__(self, cells):
        self.cells = cells
        self.dimension = len(cells)
        self.row = 0
        self.column = 0

    @staticmethod
    def fromIndex(index):
        if index == 0: # O
            cells = np.array([[0x0000AA, 0x0000AA],
                              [0x0000AA, 0x0000AA]])
        elif index == 1: # J
            cells = np.array([[0xC0C0C0, 0x000000, 0x000000],
                              [0xC0C0C0, 0xC0C0C0, 0xC0C0C0],
                              [0x000000, 0x000000, 0x000000]])
        elif index == 2: # L
            cells = np.array([[0x000000, 0x000000, 0xAA00AA],
                              [0xAA00AA, 0xAA00AA, 0xAA00AA],
                              [0x000000, 0x000000, 0x000000]])
        elif index == 3: # Z
            cells = np.array([[0x00AAAA, 0x00AAAA, 0x000000],
                              [0x000000, 0x00AAAA, 0x00AAAA],
                              [0x000000, 0x000000, 0x000000]])
        elif index == 4: # S
            cells = np.array([[0x000000, 0x00AA00, 0x00AA00],
                              [0x00AA00, 0x00AA00, 0x000000],
                              [0x000000, 0x000000, 0x000000]])
        elif index == 5: # T
            cells = np.array([[0x000000, 0xAA5500, 0x000000],
                              [0xAA5500, 0xAA5500, 0xAA5500],
                              [0x000000, 0x000000, 0x000000]])
        elif index == 6: # I
            cells = np.array([[0x000000, 0x000000, 0x000000, 0x000000],
                              [0xAA0000, 0xAA0000, 0xAA0000, 0xAA0000],
                              [0x000000, 0x000000, 0x000000, 0x000000],
                              [0x000000, 0x000000, 0x000000, 0x000000]])
        piece = Piece(cells)
        piece.row = 0
        piece.column = (10 - piece.dimension) // 2
        return piece

    def clone(self):
        _cells = np.copy(self.cells)
        piece = Piece(_cells)
        piece.row = self.row
        piece.column = self.column
        return piece
    
    def canMoveLeft(self, grid):
        for r in range(self.dimension):
            for c in range(self.dimension):
                _r = self.row + r
                _c = self.column + c - 1
                if self.cells[r][c] != 0:
                    if not (_c >= 0 and grid.cells[_r][_c] == 0):
                        return False
        return True
    
    def canMoveRight(self, grid):
        for r in range(self.dimension):
            for c in range(self.dimension):
                _r = self.row + r
                _c = self.column + c + 1
                if self.cells[r][c] != 0:
                    if not (_c >= 0 and grid.cells[_r][_c] == 0):
                        return False
        return True
    
    def canMoveDown(self, grid):
        for r in range(self.dimension):
            for c in range(self.dimension):
                _r = self.row + r + 1
                _c = self.column + c
                if self.cells[r][c] != 0 and _r >= 0:
                    if not (_r < grid.rows and grid.cells[_r][_c] == 0):
                        return False
        return True
    
    def moveLeft(self, grid):
        if not self.canMoveLeft(grid):
            return False
        self.column -= 1
        return True
    
    def moveRight(self, grid):
        if not self.canMoveRight(grid):
            return False
        self.column += 1
        return True
    
    def moveDown(self, grid):
        if not self.canMoveDown(grid):
            return False
        self.row += 1
        return True
    
    def rotateCells(self):
        _cells = np.zeros((self.dimension, self.dimension))
        if self.dimension == 2:
            _cells[0][0] = self.cells[1][0]
            _cells[0][1] = self.cells[0][0]
            _cells[1][0] = self.cells[1][1]
            _cells[1][1] = self.cells[0][1]
        elif self.dimension == 3:
            _cells[0][0] = self.cells[2][0]
            _cells[0][1] = self.cells[1][0]
            _cells[0][2] = self.cells[0][0]
            _cells[1][0] = self.cells[2][1]
            _cells[1][1] = self.cells[1][1]
            _cells[1][2] = self.cells[0][1]
            _cells[2][0] = self.cells[2][2]
            _cells[2][1] = self.cells[1][2]
            _cells[2][2] = self.cells[0][2]
        elif self.dimension == 4:
            _cells[0][0] = self.cells[3][0]
            _cells[0][1] = self.cells[2][0]
            _cells[0][2] = self.cells[1][0]
            _cells[0][3] = self.cells[0][0]
            _cells[1][3] = self.cells[0][1]
            _cells[2][3] = self.cells[0][2]
            _cells[3][3] = self.cells[0][3]
            _cells[3][2] = self.cells[1][3]
            _cells[3][1] = self.cells[2][3]
            _cells[3][0] = self.cells[3][3]
            _cells[2][0] = self.cells[3][2]
            _cells[1][0] = self.cells[3][1]
            _cells[1][1] = self.cells[2][1]
            _cells[1][2] = self.cells[1][1]
            _cells[2][2] = self.cells[1][2]
            _cells[2][1] = self.cells[2][2]

        self.cells = _cells

    def computeRotateOffset(self, grid):
        _piece = self.clone()
        _piece.rotateCells()
        if grid.valid(_piece):
            return {'rowOffset': _piece.row - self.row, 'columnOffset': _piece.column - self.column}

        initialRow = _piece.row
        initialCol = _piece.column

        for i in range(_piece.dimension - 1):
            _piece.column = initialCol + i
            if grid.valid(_piece):
                return {'rowOffset': _piece.row - self.row, 'columnOffset': _piece.column - self.column}

            for j in range(_piece.dimension - 1):
                _piece.row = initialRow - j
                if grid.valid(_piece):
                    return {'rowOffset': _piece.row - self.row, 'columnOffset': _piece.column - self.column}
            _piece.row = initialRow
        _piece.column = initialCol

        for i in range(_piece.dimension - 1):
            _piece.column = initialCol - i
            if grid.valid(_piece):
                return {'rowOffset': _piece.row - self.row, 'columnOffset': _piece.column - self.column}

            for j in range(_piece.dimension - 1):
                _piece.row = initialRow - j
                if grid.valid(_piece):
                    return {'rowOffset': _piece.row - self.row, 'columnOffset': _piece.column - self.column}
            _piece.row = initialRow
        _piece.column = initialCol

        return None
    
    def rotate(self, grid):
        offset = self.computeRotateOffset(grid)
        if offset is not None:
            self.rotateCells()
            self.row += offset['rowOffset']
            self.column += offset['columnOffset']

heightWeight = 0.510066
linesWeight = 0.760666
holesWeight = 0.35663
bumpinessWeight = 0.184483


def _best(grid, workingPieces, workingPieceIndex):
    # print("start")
    best = None
    bestScore = None
    workingPiece = workingPieces[workingPieceIndex]

    bestMoves = ""
    for rotation in range(4):
        moves = ""
        _piece = workingPiece.clone()
        for i in range(rotation):
            _piece.rotate(grid)
            moves += "T"

        while _piece.moveLeft(grid):
            moves += "L"

        while grid.valid(_piece):
            _pieceSet = _piece.clone()
            while _pieceSet.moveDown(grid):
                pass

            _grid = grid.clone()
            _grid.add_piece(_pieceSet)

            score = None
            rec_moves = ""
            if workingPieceIndex == (len(workingPieces) - 1):
                score = -heightWeight * _grid.aggregate_height() + linesWeight * _grid.lines() - holesWeight * _grid.holes() - bumpinessWeight * _grid.bumpiness()
                rec_moves = moves
            else:
                res = _best(_grid, workingPieces, workingPieceIndex + 1)
                # score = res.score
                score = res["score"]
                # rec_moves = res.moves
                rec_moves = res["moves"]

            # if score > bestScore or bestScore is None:
            if bestScore is None or score > bestScore:
                bestScore = score
                best = _piece.clone()
                bestMoves = rec_moves

            _piece.column += 1
            moves += "R"

    return {"piece": best, "score": bestScore, "moves": bestMoves}

def best(grid, workingPieces):
    res = _best(grid, workingPieces, 0)
    # print(res["moves"])
    # return res["piece"]
    return res

import numpy as np
import cv2

def hex2rgb(hex):
    hex = int(hex)
    red = (hex >> 16) & 0xff
    green = (hex >> 8) & 0xff
    blue = hex & 0xff
    return red, green, blue

def disp_board(board):
    h, w = board.shape
    side = 20
    img = np.zeros((h*side, w*side, 3), np.uint8)
    # print(board)

    for i in range(h):
        for j in range(w):
            # r,g,b = cv2.hex
            # convert hex to rgb. board contains hex values
            r, g, b = hex2rgb(board[i, j])
            cv2.rectangle(img, (j*side, i*side), (j*side+side, i*side+side), (r, g, b), -1)
    return img

bag = np.random.permutation(7)
picked = 0
def random_piece():
    # return piece.Piece.fromIndex(np.random.randint(0, 7))
    global bag
    global picked
    if picked == 7:
        picked = 0
        bag = np.random.permutation(7)
    picked += 1
    return Piece.fromIndex(bag[picked-1])
    
workingPieces = [None, random_piece()]

# class GameMaster:
#     def __init__(self, rows=20, columns=10):
#         self.grid = Grid(rows, columns)
#         self.workingPieces = [None, random_piece()]
#         self.workingPiece = None

#     def start_turn(self):
#         self.workingPieces[0] = self.workingPieces[1]
#         self.workingPieces[1] = random_piece()
#         self.workingPiece = best(self.grid, [self.workingPieces[0]])
#         while self.workingPiece.moveDown(self.grid):
#             pass
#         if not self.end_turn():
#             print("Game Over!")
#             return False
#         return True

#     def end_turn(self):
#         self.grid.add_piece(self.workingPiece)
#         # print(self.grid.cells)
#         # print(self.work
#         score = self.grid.clear_lines()
#         return not self.grid.exceeded()
    
class GameMaster:
    def __init__(self, rows=20, columns=10):
        self.grid = Grid(rows, columns)

    def start_turn(self, workingPiece, grid, disp=False):
        self.grid.cells = grid
        return best(self.grid, [workingPiece])
    
    def place_piece(self, piece):
        # takes a piece at the top of the board and places it at the bottom
        while piece.moveDown(self.grid):
            pass
        self.grid.add_piece(piece)
        score = self.grid.clear_lines()
        # return disp_board(self.grid.cells)
        return self.grid.cells

