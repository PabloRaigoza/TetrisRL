# This also exists
# https://github.com/vietnh1009/Tetris-deep-Q-learning-pytorch

# venv/lib/python3.10/site-packages/tetris_gymnasium/envs/tetris.py
# line 576 = active_tetromino_mask[active_tetromino_slices] = self.active_tetromino.matrix
import numpy as np
import cv2
import os
import execjs

js_code = """
function Grid(rows, columns, acells){
    this.rows = rows;
    this.columns = columns;

    
    this.cells = new Array(rows);
        for (var r = 0; r < rows; r++) {
            this.cells[r] = new Array(columns);
            for(var c = 0; c < columns; c++){
                //this.cells[r][c] = cells[r * columns + c];
                if (acells[r * columns + c] == 0) this.cells[r][c] = 0x000000;
                else if (acells[r * columns + c] == 1) this.cells[r][c] = 0x000000;
                else if (acells[r * columns + c] == 2) this.cells[r][c] = 0xAA0000;
                else if (acells[r * columns + c] == 3) this.cells[r][c] = 0x0000AA;
                else if (acells[r * columns + c] == 4) this.cells[r][c] = 0xAA5500;
                else if (acells[r * columns + c] == 5) this.cells[r][c] = 0x00AA00;
                else if (acells[r * columns + c] == 6) this.cells[r][c] = 0x00AAAA;
                else if (acells[r * columns + c] == 7) this.cells[r][c] = 0x5500AA;
                else if (acells[r * columns + c] == 8) this.cells[r][c] = 0xAA00AA;
            }
        }
};

Grid.prototype.clone = function(){
    var _grid = new Grid(this.rows, this.columns);
    for (var r = 0; r < this.rows; r++) {
        for(var c = 0; c < this.columns; c++){
            _grid.cells[r][c] = this.cells[r][c];
        }
    }
    return _grid;
};

Grid.prototype.clearLines = function(){
    var distance = 0;
    var row = new Array(this.columns);
    for(var r = this.rows - 1; r >= 0; r--){
        if (this.isLine(r)){
            distance++;
            for(var c = 0; c < this.columns; c++){
                this.cells[r][c] = 0;
            }
        }else if (distance > 0){
            for(var c = 0; c < this.columns; c++){
                this.cells[r + distance][c] = this.cells[r][c];
                this.cells[r][c] = 0;
            }
        }
    }
    return distance;
};

Grid.prototype.isLine = function(row){
    for(var c = 0; c < this.columns; c++){
        if (this.cells[row][c] == 0){
            return false;
        }
    }
    return true;
};

Grid.prototype.isEmptyRow = function(row){
    for(var c = 0; c < this.columns; c++){
        if (this.cells[row][c] != 0){
            return false;
        }
    }
    return true;
};

Grid.prototype.exceeded = function(){
    return !this.isEmptyRow(0) || !this.isEmptyRow(1);
};

Grid.prototype.height = function(){
    var r = 0;
    for(; r < this.rows && this.isEmptyRow(r); r++);
    return this.rows - r;
};

Grid.prototype.lines = function(){
    var count = 0;
    for(var r = 0; r < this.rows; r++){
        if (this.isLine(r)){
            count++;
        }
    }
    return count;
};

Grid.prototype.holes = function(){
    var count = 0;
    for(var c = 0; c < this.columns; c++){
        var block = false;
        for(var r = 0; r < this.rows; r++){
            if (this.cells[r][c] != 0) {
                block = true;
            }else if (this.cells[r][c] == 0 && block){
                count++;
            }
        }
    }
    return count;
};

Grid.prototype.blockades = function(){
    var count = 0;
    for(var c = 0; c < this.columns; c++){
        var hole = false;
        for(var r = this.rows - 1; r >= 0; r--){
            if (this.cells[r][c] == 0){
                hole = true;
            }else if (this.cells[r][c] != 0 && hole){
                count++;
            }
        }
    }
    return count;
}

Grid.prototype.aggregateHeight = function(){
    var total = 0;
    for(var c = 0; c < this.columns; c++){
        total += this.columnHeight(c);
    }
    return total;
};

Grid.prototype.bumpiness = function(){
    var total = 0;
    for(var c = 0; c < this.columns - 1; c++){
        total += Math.abs(this.columnHeight(c) - this.columnHeight(c+ 1));
    }
    return total;
}

Grid.prototype.columnHeight = function(column){
    var r = 0;
    for(; r < this.rows && this.cells[r][column] == 0; r++);
    return this.rows - r;
};

Grid.prototype.addPiece = function(piece) {
    for(var r = 0; r < piece.cells.length; r++) {
        for (var c = 0; c < piece.cells[r].length; c++) {
            var _r = piece.row + r;
            var _c = piece.column + c;
            if (piece.cells[r][c] != 0 && _r >= 0){
                this.cells[_r][_c] = piece.cells[r][c];
            }
        }
    }
};

Grid.prototype.valid = function(piece){
    for(var r = 0; r < piece.cells.length; r++){
        for(var c = 0; c < piece.cells[r].length; c++){
            var _r = piece.row + r;
            var _c = piece.column + c;
            if (piece.cells[r][c] != 0){
                if(_r < 0 || _r >= this.rows){
                    return false;
                }
                if(_c < 0 || _c >= this.columns){
                    return false;
                }
                if (this.cells[_r][_c] != 0){
                    return false;
                }
            }
        }
    }
    return true;
};

function Piece(cells, row, column){
    this.cells = cells;

    this.dimension = this.cells.length;
    // this.row = 0;
    // this.column = 0;
    if (row == null){
        this.row = 0;
    }else{
        this.row = row;
    }

    if (column == null){
        this.column = 0;
    }else{
        this.column = column;
    }
};

Piece.fromIndex = function(index){
    var piece;
    switch (index){
        case 0:// O
            piece = new Piece([
                [0x0000AA, 0x0000AA],
                [0x0000AA, 0x0000AA]
            ]);
            break;
        case 1: // J
            piece = new Piece([
                [0xC0C0C0, 0x000000, 0x000000],
                [0xC0C0C0, 0xC0C0C0, 0xC0C0C0],
                [0x000000, 0x000000, 0x000000]
            ]);
            break;
        case 2: // L
            piece = new Piece([
                [0x000000, 0x000000, 0xAA00AA],
                [0xAA00AA, 0xAA00AA, 0xAA00AA],
                [0x000000, 0x000000, 0x000000]
            ]);
            break;
        case 3: // Z
            piece = new Piece([
                [0x00AAAA, 0x00AAAA, 0x000000],
                [0x000000, 0x00AAAA, 0x00AAAA],
                [0x000000, 0x000000, 0x000000]
            ]);
            break;
        case 4: // S
            piece = new Piece([
                [0x000000, 0x00AA00, 0x00AA00],
                [0x00AA00, 0x00AA00, 0x000000],
                [0x000000, 0x000000, 0x000000]
            ]);
            break;
        case 5: // T
            piece = new Piece([
                [0x000000, 0xAA5500, 0x000000],
                [0xAA5500, 0xAA5500, 0xAA5500],
                [0x000000, 0x000000, 0x000000]
            ]);
            break;
        case 6: // I
            piece = new Piece([
                [0x000000, 0x000000, 0x000000, 0x000000],
                [0xAA0000, 0xAA0000, 0xAA0000, 0xAA0000],
                [0x000000, 0x000000, 0x000000, 0x000000],
                [0x000000, 0x000000, 0x000000, 0x000000]
            ]);
            break;

    }
    piece.row = 0;
    piece.column = Math.floor((10 - piece.dimension) / 2); // Centralize
    return piece;
};

Piece.prototype.clone = function(){
    var _cells = new Array(this.dimension);
    for (var r = 0; r < this.dimension; r++) {
        _cells[r] = new Array(this.dimension);
        for(var c = 0; c < this.dimension; c++){
            _cells[r][c] = this.cells[r][c];
        }
    }

    var piece = new Piece(_cells);
    piece.row = this.row;
    piece.column = this.column;
    return piece;
};

Piece.prototype.canMoveLeft = function(grid){
    for(var r = 0; r < this.cells.length; r++){
        for(var c = 0; c < this.cells[r].length; c++){
            var _r = this.row + r;
            var _c = this.column + c - 1;
            if (this.cells[r][c] != 0){
                if (!(_c >= 0 && grid.cells[_r][_c] == 0)){
                    return false;
                }
            }
        }
    }
    return true;
};

Piece.prototype.canMoveRight = function(grid){
    for(var r = 0; r < this.cells.length; r++){
        for(var c = 0; c < this.cells[r].length; c++){
            var _r = this.row + r;
            var _c = this.column + c + 1;
            if (this.cells[r][c] != 0){
                if (!(_c >= 0 && grid.cells[_r][_c] == 0)){
                    return false;
                }
            }
        }
    }
    return true;
};


Piece.prototype.canMoveDown = function(grid){
    for(var r = 0; r < this.cells.length; r++){
        for(var c = 0; c < this.cells[r].length; c++){
            var _r = this.row + r + 1;
            var _c = this.column + c;
            if (this.cells[r][c] != 0 && _r >= 0){
                if (!(_r < grid.rows && grid.cells[_r][_c] == 0)){
                    return false;
                }
            }
        }
    }
    return true;
};

Piece.prototype.moveLeft = function(grid){
    if(!this.canMoveLeft(grid)){
        return false;
    }
    this.column--;
    return true;
};

Piece.prototype.moveRight = function(grid){
    if(!this.canMoveRight(grid)){
        return false;
    }
    this.column++;
    return true;
};

Piece.prototype.moveDown = function(grid){
    if(!this.canMoveDown(grid)){
        return false;
    }
    this.row++;
    return true;
};

Piece.prototype.rotateCells = function(){
      var _cells = new Array(this.dimension);
      for (var r = 0; r < this.dimension; r++) {
          _cells[r] = new Array(this.dimension);
      }

      switch (this.dimension) { // Assumed square matrix
          case 2:
              _cells[0][0] = this.cells[1][0];
              _cells[0][1] = this.cells[0][0];
              _cells[1][0] = this.cells[1][1];
              _cells[1][1] = this.cells[0][1];
              break;
          case 3:
              _cells[0][0] = this.cells[2][0];
              _cells[0][1] = this.cells[1][0];
              _cells[0][2] = this.cells[0][0];
              _cells[1][0] = this.cells[2][1];
              _cells[1][1] = this.cells[1][1];
              _cells[1][2] = this.cells[0][1];
              _cells[2][0] = this.cells[2][2];
              _cells[2][1] = this.cells[1][2];
              _cells[2][2] = this.cells[0][2];
              break;
          case 4:
              _cells[0][0] = this.cells[3][0];
              _cells[0][1] = this.cells[2][0];
              _cells[0][2] = this.cells[1][0];
              _cells[0][3] = this.cells[0][0];
              _cells[1][3] = this.cells[0][1];
              _cells[2][3] = this.cells[0][2];
              _cells[3][3] = this.cells[0][3];
              _cells[3][2] = this.cells[1][3];
              _cells[3][1] = this.cells[2][3];
              _cells[3][0] = this.cells[3][3];
              _cells[2][0] = this.cells[3][2];
              _cells[1][0] = this.cells[3][1];

              _cells[1][1] = this.cells[2][1];
              _cells[1][2] = this.cells[1][1];
              _cells[2][2] = this.cells[1][2];
              _cells[2][1] = this.cells[2][2];
              break;
      }

      this.cells = _cells;
};

Piece.prototype.computeRotateOffset = function(grid){
    var _piece = this.clone();
    _piece.rotateCells();
    if (grid.valid(_piece)) {
        return { rowOffset: _piece.row - this.row, columnOffset: _piece.column - this.column };
    }

    // Kicking
    var initialRow = _piece.row;
    var initialCol = _piece.column;

    for (var i = 0; i < _piece.dimension - 1; i++) {
        _piece.column = initialCol + i;
        if (grid.valid(_piece)) {
            return { rowOffset: _piece.row - this.row, columnOffset: _piece.column - this.column };
        }

        for (var j = 0; j < _piece.dimension - 1; j++) {
            _piece.row = initialRow - j;
            if (grid.valid(_piece)) {
                return { rowOffset: _piece.row - this.row, columnOffset: _piece.column - this.column };
            }
        }
        _piece.row = initialRow;
    }
    _piece.column = initialCol;

    for (var i = 0; i < _piece.dimension - 1; i++) {
        _piece.column = initialCol - i;
        if (grid.valid(_piece)) {
            return { rowOffset: _piece.row - this.row, columnOffset: _piece.column - this.column };
        }

        for (var j = 0; j < _piece.dimension - 1; j++) {
            _piece.row = initialRow - j;
            if (grid.valid(_piece)) {
                return { rowOffset: _piece.row - this.row, columnOffset: _piece.column - this.column };
            }
        }
        _piece.row = initialRow;
    }
    _piece.column = initialCol;

    return null;
};

Piece.prototype.rotate = function(grid){
    var offset = this.computeRotateOffset(grid);
    if (offset != null){
        this.rotateCells(grid);
        this.row += offset.rowOffset;
        this.column += offset.columnOffset;
    }
};

/**
 * @param {Object} weights
 * @param {number} weights.heightWeight
 * @param {number} weights.linesWeight
 * @param {number} weights.holesWeight
 * @param {number} weights.bumpinessWeight
 */
function AI(weights){
    this.heightWeight = weights.heightWeight;
    this.linesWeight = weights.linesWeight;
    this.holesWeight = weights.holesWeight;
    this.bumpinessWeight = weights.bumpinessWeight;
};

AI.prototype._best = function(grid, workingPieces, workingPieceIndex){
    console.log("start");
    var best = null;
    var bestScore = null;
    var workingPiece = workingPieces[workingPieceIndex];

    var bestMoves = "";
    for(var rotation = 0; rotation < 4; rotation++){
        var moves = "";
        var _piece = workingPiece.clone();
        for(var i = 0; i < rotation; i++){
            _piece.rotate(grid);
            moves+="T";
        }

        while(_piece.moveLeft(grid)) moves+="L";

        while(grid.valid(_piece)){
            var _pieceSet = _piece.clone();
            while(_pieceSet.moveDown(grid));

            var _grid = grid.clone();
            _grid.addPiece(_pieceSet);

            var score = null;
            var rec_moves = "";
            if (workingPieceIndex == (workingPieces.length - 1)) {
                score = -this.heightWeight * _grid.aggregateHeight() + this.linesWeight * _grid.lines() - this.holesWeight * _grid.holes() - this.bumpinessWeight * _grid.bumpiness();
                rec_moves = moves;
            }else{
                // score = this._best(_grid, workingPieces, workingPieceIndex + 1).score;
                var res = this._best(_grid, workingPieces, workingPieceIndex + 1);
                score = res.score;
                rec_moves = res.moves;
            }

            if (score > bestScore || bestScore == null){
                bestScore = score;
                best = _piece.clone();
                bestMoves = rec_moves;  
            }

            _piece.column++;
            moves += "R";
        }
    }

    //return {piece:best, score:bestScore, moves:moves};
    return {piece:best, score:bestScore, moves:bestMoves};
};

AI.prototype.best = function(grid, workingPieces){
    var res = this._best(grid, workingPieces, 0);
    //console.log(res.moves);
    return res;
};

function calculateBestMove(numRows, numColumns, cells, pieceIndex, pieceRow, pieceCol, nextPieceIndex, nextPieceRow, nextPieceCol, weights){
    var grid = new Grid(20, 10, cells);
    //var piece = new Piece(pieceCells, pieceRow, pieceCol);
    //var nextPiece = new Piece(nextPieceCells, nextPieceRow, nextPieceCol);
    var piece = Piece.fromIndex(pieceIndex);
    //piece.moveRight(grid);
    var nextPiece = Piece.fromIndex(nextPieceIndex);
    var ai = new AI(weights);

    // Add the piece to the grid
    var res = ai.best(grid, [piece, nextPiece]);
    
    //grid.addPiece(res.piece.clone());
    return res.moves;
}
"""

# // Count the number of rotations
#     var rotation = 0;
#     for(var i = 0; i < res.moves.length; i++){
#         if (res.moves[i] == "T"){
#             rotation = (rotation + 1) % 4;
#         }
#     }

#     var _piece = piece.clone();
#     var finalMoves = "";
#     if (_piece.column < pieceCol){
#         for(var i = 0; i < pieceCol - _piece.column; i++){
#             finalMoves += "L";
#         }
#     }else if (_piece.column > pieceCol){
#         for(var i = 0; i < _piece.column - pieceCol; i++){
#             finalMoves += "R";
#         }
#     }

#     for(var i = 0; i < rotation; i++){
#         finalMoves += "T";
#     }

ctx = execjs.compile(js_code)

# Make AI
weights = {
    'heightWeight': 0.510066,
    'linesWeight': 0.760666,
    'holesWeight': 0.35663,
    'bumpinessWeight': 0.184483
}


import data.perfect_bot as bot

from utils.environment import makeBC, MAX_STEPS


# Create data directory
data_dir = "data/BC"
os.makedirs(data_dir, exist_ok=True)


# Create data file name
num = len(os.listdir(data_dir))
seed = np.random.randint(0, 1000000)
file_name = f"BC_data_{str(num).zfill(3)}_{str(seed).zfill(6)}_{MAX_STEPS}.npy"


# Create an instance of Tetris
env = makeBC()
observation = env.reset(seed=seed)


# Initialize data array of MAX_STEPS
data = np.array([{} for _ in range(MAX_STEPS)])
terminated = False
steps = 0

def extract_piece(mat, next_piece=False):
    rows, cols = np.where(mat != 0)
    min_x, max_x = np.min(cols), np.max(cols)
    min_y, max_y = np.min(rows), np.max(rows)
    p = bot.Piece(mat[min_y:max_y+1, min_x:max_x+1])
    if next_piece:
        p.row = 0
        p.col = (10 - p.dimension) // 2
    else:
        p.row = min_y
        p.col = min_x
    return p

# Collect data loop
while not terminated and steps < MAX_STEPS:
    # Render the current state of the game
    env.render()

    # print(observation[0])
    cur_board = observation[0]['board'][:20, 4:14] if steps == 0 else observation['board'][:20, 4:14]
    cur_next_piece = observation[0]['queue'][:,:4] if steps == 0 else observation['queue'][:,:4]
    cur_active_piece = observation[0]['active_tetromino_mask'][:20,4:14] if steps == 0 else observation['active_tetromino_mask'][:20,4:14]
    # print(cur_board)
    # print(observation[0]['active_tetromino_mask'] if steps == 0 else observation['active_tetromino_mask'])
    # I 2 -> 0xAA0000 6
    # O 3 -> 0x0000AA 0
    # T 4 -> 0xAA5500 5
    # S 5 -> 0x00AA00 4
    # Z 6 -> 0x00AAAA 3
    # J 7 -> 0x5500AA 1
    # L 8 -> 0xAA00AA 2

    hex_mappings = {
        0: 0x000000,
        1: 0x000000,
        2: 0xAA0000,
        3: 0x0000AA,
        4: 0xAA5500,
        5: 0x00AA00,
        6: 0x00AAAA,
        7: 0x5500AA,
        8: 0xAA00AA
    }

    index_mappings = [0,0,6,0,5,4,3,1,2]

    my_grid = bot.Grid(cur_board.shape[1], cur_board.shape[0])
    my_grid.grid = cur_board
    my_piece = extract_piece(cur_active_piece)
    my_next_piece = extract_piece(cur_next_piece, next_piece=True)

    # js_grid = ctx.call("Grid", my_grid.rows, my_grid.columns)
    # js_grid.set_cell(my_grid.grid.tolist())
    
    # js_grid = ctx.call("Grid", my_grid.rows, my_grid.columns, my_grid.grid.tolist())
    # js_piece = ctx.call("Piece", my_piece.cells.tolist(), int(my_piece.row), int(my_piece.col))
    # js_next_piece = ctx.call("Piece", my_next_piece.cells.tolist(), int(my_next_piece.row), int(my_next_piece.col))
    
    print(my_piece.row, my_piece.col)
    my_piece_index = index_mappings[np.argmax(my_piece.cells)]
    my_next_piece_index = index_mappings[np.argmax(my_next_piece.cells)]
    # best_move = ctx.call("calculateBestMove", my_grid.rows, my_grid.columns, my_grid.grid.tolist(), my_piece.cells.tolist(), int(my_piece.row), int(my_piece.col), my_next_piece.cells.tolist(), int(my_next_piece.row), int(my_next_piece.col), weights)
    grid_string = ""
    # for row in my_grid.grid:
    #     for cell in row:
    #         grid_string += str(cur_board[cell])

    for row in range(0, 20):
        for col in range(0, 10):
            grid_string += str(cur_board[row, col])
    for row in range(0, 20):
        print(grid_string[row*10:row*10+10])
    # print(my_grid.grid.tolist())
    # grid_cells = my_grid.grid.tolist()
    # for ro in range(0, 20):
    #     for co in range(0, 10):
    #         grid_cells[ro][co] = hex_mappings[grid_cells[ro][co]]
    # print(grid_cells)
    best_move = ctx.call("calculateBestMove", 20, 10, grid_string, my_piece_index, int(my_piece.row), int(my_piece.col), my_next_piece_index, int(my_next_piece.row), int(my_next_piece.col), weights)
    # best_move = ctx.call("calculateBestMove", 20, 10, grid_cells, my_piece_index, int(my_piece.row), int(my_piece.col), my_next_piece_index, int(my_next_piece.row), int(my_next_piece.col), weights)
    print(best_move)
    # best_move = "T" * best_move

    # best_move, row, col, score = best_move.split(",")
    # print(int(row), int(col), float(score))

    # final_moves = ""
    # if int(col) < my_piece.col: final_moves += "L" * (my_piece.col - int(col))
    # elif int(col) > my_piece.col: final_moves += "R" * (int(col) - my_piece.col)
    # t_moves = 0
    # for letter in best_move:
    #     if letter == "T": t_moves = (t_moves + 1) % 4
    # final_moves += "T" * t_moves
    # print(final_moves)
    for move in best_move:
        if move == "L":
            action = env.unwrapped.actions.move_left
        elif move == "R":
            action = env.unwrapped.actions.move_right
        elif move == "T":
            action = env.unwrapped.actions.rotate_counterclockwise
        obs, reward, terminated, truncated, info = env.step(action)
        data[steps] = {
            'state': observation,
            'action': action,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
            'info': info
        }
        observation = obs
        steps += 1
        env.render()
        cv2.waitKey(1000)
    action = env.unwrapped.actions.hard_drop
    obs, reward, terminated, truncated, info = env.step(action)
    observation = obs
    steps += 1
    env.render()
    cv2.waitKey(1000)

# Close the environment
print(f"Data saved to {os.path.join(data_dir, file_name)}")
env.close()
cv2.destroyAllWindows()


# LLLLRRRRRRRRRTLLLLRRRRRRRRRTTLLLLRRRRRRRRRTTTLLLLRRRRRRRRR
