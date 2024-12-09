# function Grid(rows, columns){
#     this.rows = rows;
#     this.columns = columns;

#     this.cells = new Array(rows);
#     for (var r = 0; r < this.rows; r++) {
#         this.cells[r] = new Array(this.columns);
#         for(var c = 0; c < this.columns; c++){
#             this.cells[r][c] = 0;
#         }
#     }
# };

# Grid.prototype.clone = function(){
#     var _grid = new Grid(this.rows, this.columns);
#     for (var r = 0; r < this.rows; r++) {
#         for(var c = 0; c < this.columns; c++){
#             _grid.cells[r][c] = this.cells[r][c];
#         }
#     }
#     return _grid;
# };

# Grid.prototype.clearLines = function(){
#     var distance = 0;
#     var row = new Array(this.columns);
#     for(var r = this.rows - 1; r >= 0; r--){
#         if (this.isLine(r)){
#             distance++;
#             for(var c = 0; c < this.columns; c++){
#                 this.cells[r][c] = 0;
#             }
#         }else if (distance > 0){
#             for(var c = 0; c < this.columns; c++){
#                 this.cells[r + distance][c] = this.cells[r][c];
#                 this.cells[r][c] = 0;
#             }
#         }
#     }
#     return distance;
# };

# Grid.prototype.isLine = function(row){
#     for(var c = 0; c < this.columns; c++){
#         if (this.cells[row][c] == 0){
#             return false;
#         }
#     }
#     return true;
# };

# Grid.prototype.isEmptyRow = function(row){
#     for(var c = 0; c < this.columns; c++){
#         if (this.cells[row][c] != 0){
#             return false;
#         }
#     }
#     return true;
# };

# Grid.prototype.exceeded = function(){
#     return !this.isEmptyRow(0) || !this.isEmptyRow(1);
# };

# Grid.prototype.height = function(){
#     var r = 0;
#     for(; r < this.rows && this.isEmptyRow(r); r++);
#     return this.rows - r;
# };

# Grid.prototype.lines = function(){
#     var count = 0;
#     for(var r = 0; r < this.rows; r++){
#         if (this.isLine(r)){
#             count++;
#         }
#     }
#     return count;
# };

# Grid.prototype.holes = function(){
#     var count = 0;
#     for(var c = 0; c < this.columns; c++){
#         var block = false;
#         for(var r = 0; r < this.rows; r++){
#             if (this.cells[r][c] != 0) {
#                 block = true;
#             }else if (this.cells[r][c] == 0 && block){
#                 count++;
#             }
#         }
#     }
#     return count;
# };

# Grid.prototype.blockades = function(){
#     var count = 0;
#     for(var c = 0; c < this.columns; c++){
#         var hole = false;
#         for(var r = this.rows - 1; r >= 0; r--){
#             if (this.cells[r][c] == 0){
#                 hole = true;
#             }else if (this.cells[r][c] != 0 && hole){
#                 count++;
#             }
#         }
#     }
#     return count;
# }

# Grid.prototype.aggregateHeight = function(){
#     var total = 0;
#     for(var c = 0; c < this.columns; c++){
#         total += this.columnHeight(c);
#     }
#     return total;
# };

# Grid.prototype.bumpiness = function(){
#     var total = 0;
#     for(var c = 0; c < this.columns - 1; c++){
#         total += Math.abs(this.columnHeight(c) - this.columnHeight(c+ 1));
#     }
#     return total;
# }

# Grid.prototype.columnHeight = function(column){
#     var r = 0;
#     for(; r < this.rows && this.cells[r][column] == 0; r++);
#     return this.rows - r;
# };

# Grid.prototype.addPiece = function(piece) {
#     for(var r = 0; r < piece.cells.length; r++) {
#         for (var c = 0; c < piece.cells[r].length; c++) {
#             var _r = piece.row + r;
#             var _c = piece.column + c;
#             if (piece.cells[r][c] != 0 && _r >= 0){
#                 this.cells[_r][_c] = piece.cells[r][c];
#             }
#         }
#     }
# };

# Grid.prototype.valid = function(piece){
#     for(var r = 0; r < piece.cells.length; r++){
#         for(var c = 0; c < piece.cells[r].length; c++){
#             var _r = piece.row + r;
#             var _c = piece.column + c;
#             if (piece.cells[r][c] != 0){
#                 if(_r < 0 || _r >= this.rows){
#                     return false;
#                 }
#                 if(_c < 0 || _c >= this.columns){
#                     return false;
#                 }
#                 if (this.cells[_r][_c] != 0){
#                     return false;
#                 }
#             }
#         }
#     }
#     return true;
# };

# function Piece(cells){
#     this.cells = cells;

#     this.dimension = this.cells.length;
#     this.row = 0;
#     this.column = 0;
# };

# Piece.fromIndex = function(index){
#     var piece;
#     switch (index){
#         case 0:// O
#             piece = new Piece([
#                 [0x0000AA, 0x0000AA],
#                 [0x0000AA, 0x0000AA]
#             ]);
#             break;
#         case 1: // J
#             piece = new Piece([
#                 [0xC0C0C0, 0x000000, 0x000000],
#                 [0xC0C0C0, 0xC0C0C0, 0xC0C0C0],
#                 [0x000000, 0x000000, 0x000000]
#             ]);
#             break;
#         case 2: // L
#             piece = new Piece([
#                 [0x000000, 0x000000, 0xAA00AA],
#                 [0xAA00AA, 0xAA00AA, 0xAA00AA],
#                 [0x000000, 0x000000, 0x000000]
#             ]);
#             break;
#         case 3: // Z
#             piece = new Piece([
#                 [0x00AAAA, 0x00AAAA, 0x000000],
#                 [0x000000, 0x00AAAA, 0x00AAAA],
#                 [0x000000, 0x000000, 0x000000]
#             ]);
#             break;
#         case 4: // S
#             piece = new Piece([
#                 [0x000000, 0x00AA00, 0x00AA00],
#                 [0x00AA00, 0x00AA00, 0x000000],
#                 [0x000000, 0x000000, 0x000000]
#             ]);
#             break;
#         case 5: // T
#             piece = new Piece([
#                 [0x000000, 0xAA5500, 0x000000],
#                 [0xAA5500, 0xAA5500, 0xAA5500],
#                 [0x000000, 0x000000, 0x000000]
#             ]);
#             break;
#         case 6: // I
#             piece = new Piece([
#                 [0x000000, 0x000000, 0x000000, 0x000000],
#                 [0xAA0000, 0xAA0000, 0xAA0000, 0xAA0000],
#                 [0x000000, 0x000000, 0x000000, 0x000000],
#                 [0x000000, 0x000000, 0x000000, 0x000000]
#             ]);
#             break;

#     }
#     piece.row = 0;
#     piece.column = Math.floor((10 - piece.dimension) / 2); // Centralize
#     return piece;
# };

# Piece.prototype.clone = function(){
#     var _cells = new Array(this.dimension);
#     for (var r = 0; r < this.dimension; r++) {
#         _cells[r] = new Array(this.dimension);
#         for(var c = 0; c < this.dimension; c++){
#             _cells[r][c] = this.cells[r][c];
#         }
#     }

#     var piece = new Piece(_cells);
#     piece.row = this.row;
#     piece.column = this.column;
#     return piece;
# };

# Piece.prototype.canMoveLeft = function(grid){
#     for(var r = 0; r < this.cells.length; r++){
#         for(var c = 0; c < this.cells[r].length; c++){
#             var _r = this.row + r;
#             var _c = this.column + c - 1;
#             if (this.cells[r][c] != 0){
#                 if (!(_c >= 0 && grid.cells[_r][_c] == 0)){
#                     return false;
#                 }
#             }
#         }
#     }
#     return true;
# };

# Piece.prototype.canMoveRight = function(grid){
#     for(var r = 0; r < this.cells.length; r++){
#         for(var c = 0; c < this.cells[r].length; c++){
#             var _r = this.row + r;
#             var _c = this.column + c + 1;
#             if (this.cells[r][c] != 0){
#                 if (!(_c >= 0 && grid.cells[_r][_c] == 0)){
#                     return false;
#                 }
#             }
#         }
#     }
#     return true;
# };


# Piece.prototype.canMoveDown = function(grid){
#     for(var r = 0; r < this.cells.length; r++){
#         for(var c = 0; c < this.cells[r].length; c++){
#             var _r = this.row + r + 1;
#             var _c = this.column + c;
#             if (this.cells[r][c] != 0 && _r >= 0){
#                 if (!(_r < grid.rows && grid.cells[_r][_c] == 0)){
#                     return false;
#                 }
#             }
#         }
#     }
#     return true;
# };

# Piece.prototype.moveLeft = function(grid){
#     if(!this.canMoveLeft(grid)){
#         return false;
#     }
#     this.column--;
#     return true;
# };

# Piece.prototype.moveRight = function(grid){
#     if(!this.canMoveRight(grid)){
#         return false;
#     }
#     this.column++;
#     return true;
# };

# Piece.prototype.moveDown = function(grid){
#     if(!this.canMoveDown(grid)){
#         return false;
#     }
#     this.row++;
#     return true;
# };

# Piece.prototype.rotateCells = function(){
#       var _cells = new Array(this.dimension);
#       for (var r = 0; r < this.dimension; r++) {
#           _cells[r] = new Array(this.dimension);
#       }

#       switch (this.dimension) { // Assumed square matrix
#           case 2:
#               _cells[0][0] = this.cells[1][0];
#               _cells[0][1] = this.cells[0][0];
#               _cells[1][0] = this.cells[1][1];
#               _cells[1][1] = this.cells[0][1];
#               break;
#           case 3:
#               _cells[0][0] = this.cells[2][0];
#               _cells[0][1] = this.cells[1][0];
#               _cells[0][2] = this.cells[0][0];
#               _cells[1][0] = this.cells[2][1];
#               _cells[1][1] = this.cells[1][1];
#               _cells[1][2] = this.cells[0][1];
#               _cells[2][0] = this.cells[2][2];
#               _cells[2][1] = this.cells[1][2];
#               _cells[2][2] = this.cells[0][2];
#               break;
#           case 4:
#               _cells[0][0] = this.cells[3][0];
#               _cells[0][1] = this.cells[2][0];
#               _cells[0][2] = this.cells[1][0];
#               _cells[0][3] = this.cells[0][0];
#               _cells[1][3] = this.cells[0][1];
#               _cells[2][3] = this.cells[0][2];
#               _cells[3][3] = this.cells[0][3];
#               _cells[3][2] = this.cells[1][3];
#               _cells[3][1] = this.cells[2][3];
#               _cells[3][0] = this.cells[3][3];
#               _cells[2][0] = this.cells[3][2];
#               _cells[1][0] = this.cells[3][1];

#               _cells[1][1] = this.cells[2][1];
#               _cells[1][2] = this.cells[1][1];
#               _cells[2][2] = this.cells[1][2];
#               _cells[2][1] = this.cells[2][2];
#               break;
#       }

#       this.cells = _cells;
# };

# Piece.prototype.computeRotateOffset = function(grid){
#     var _piece = this.clone();
#     _piece.rotateCells();
#     if (grid.valid(_piece)) {
#         return { rowOffset: _piece.row - this.row, columnOffset: _piece.column - this.column };
#     }

#     // Kicking
#     var initialRow = _piece.row;
#     var initialCol = _piece.column;

#     for (var i = 0; i < _piece.dimension - 1; i++) {
#         _piece.column = initialCol + i;
#         if (grid.valid(_piece)) {
#             return { rowOffset: _piece.row - this.row, columnOffset: _piece.column - this.column };
#         }

#         for (var j = 0; j < _piece.dimension - 1; j++) {
#             _piece.row = initialRow - j;
#             if (grid.valid(_piece)) {
#                 return { rowOffset: _piece.row - this.row, columnOffset: _piece.column - this.column };
#             }
#         }
#         _piece.row = initialRow;
#     }
#     _piece.column = initialCol;

#     for (var i = 0; i < _piece.dimension - 1; i++) {
#         _piece.column = initialCol - i;
#         if (grid.valid(_piece)) {
#             return { rowOffset: _piece.row - this.row, columnOffset: _piece.column - this.column };
#         }

#         for (var j = 0; j < _piece.dimension - 1; j++) {
#             _piece.row = initialRow - j;
#             if (grid.valid(_piece)) {
#                 return { rowOffset: _piece.row - this.row, columnOffset: _piece.column - this.column };
#             }
#         }
#         _piece.row = initialRow;
#     }
#     _piece.column = initialCol;

#     return null;
# };

# Piece.prototype.rotate = function(grid){
#     var offset = this.computeRotateOffset(grid);
#     if (offset != null){
#         this.rotateCells(grid);
#         this.row += offset.rowOffset;
#         this.column += offset.columnOffset;
#     }
# };

# /**
#  * @param {Object} weights
#  * @param {number} weights.heightWeight
#  * @param {number} weights.linesWeight
#  * @param {number} weights.holesWeight
#  * @param {number} weights.bumpinessWeight
#  */
# function AI(weights){
#     this.heightWeight = weights.heightWeight;
#     this.linesWeight = weights.linesWeight;
#     this.holesWeight = weights.holesWeight;
#     this.bumpinessWeight = weights.bumpinessWeight;
# };

# AI.prototype._best = function(grid, workingPieces, workingPieceIndex){
#     var best = null;
#     var bestScore = null;
#     var workingPiece = workingPieces[workingPieceIndex];

#     for(var rotation = 0; rotation < 4; rotation++){
#         var _piece = workingPiece.clone();
#         for(var i = 0; i < rotation; i++){
#             _piece.rotate(grid);
#         }

#         while(_piece.moveLeft(grid));

#         while(grid.valid(_piece)){
#             var _pieceSet = _piece.clone();
#             while(_pieceSet.moveDown(grid));

#             var _grid = grid.clone();
#             _grid.addPiece(_pieceSet);

#             var score = null;
#             if (workingPieceIndex == (workingPieces.length - 1)) {
#                 score = -this.heightWeight * _grid.aggregateHeight() + this.linesWeight * _grid.lines() - this.holesWeight * _grid.holes() - this.bumpinessWeight * _grid.bumpiness();
#             }else{
#                 score = this._best(_grid, workingPieces, workingPieceIndex + 1).score;
#             }

#             if (score > bestScore || bestScore == null){
#                 bestScore = score;
#                 best = _piece.clone();
#             }

#             _piece.column++;
#         }
#     }

#     return {piece:best, score:bestScore};
# };

# AI.prototype.best = function(grid, workingPieces){
#     var res = this._best(grid, workingPieces, 0);
#     console.log(res);
#     return res.piece;
# };



import numpy as np
class Grid:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.cells = np.zeros((rows, columns), dtype=np.int32)

    def clone(self):
        _grid = Grid(self.rows, self.columns)
        _grid.cells = np.copy(self.cells)
        return _grid
    
    def clearLines(self):
        distance = 0
        for r in range(self.rows - 1, -1, -1):
            if self.isLine(r):
                distance += 1
                self.cells[r] = np.zeros(self.columns, dtype=np.int32)
            elif distance > 0:
                self.cells[r + distance] = self.cells[r]
                self.cells[r] = np.zeros(self.columns, dtype=np.int32)
        return distance
    
    def isLine(self, row):
        return np.all(self.cells[row] != 0)
    
    def isEmptyRow(self, row):
        return np.all(self.cells[row] == 0)
    
    def exceeded(self):
        return not self.isEmptyRow(0) or not self.isEmptyRow(1)
    
    def height(self):
        r = 0
        while r < self.rows and self.isEmptyRow(r):
            r += 1
        return self.rows - r
    
    def lines(self):
        count = 0
        for r in range(self.rows):
            if self.isLine(r):
                count += 1
        return count
    
    def holes(self):
        count = 0
        for c in range(self.columns):
            block = False
            for r in range(self.rows):
                if self.cells[r, c] != 0:
                    block = True
                elif self.cells[r, c] == 0 and block:
                    count += 1
        return count
    
    def blockades(self):
        count = 0
        for c in range(self.columns):
            hole = False
            for r in range(self.rows - 1, -1, -1):
                if self.cells[r, c] == 0:
                    hole = True
                elif self.cells[r, c] != 0 and hole:
                    count += 1
        return count
    
    def aggregateHeight(self):
        total = 0
        for c in range(self.columns):
            total += self.columnHeight(c)
        return total
    
    def bumpiness(self):
        total = 0
        for c in range(self.columns - 1):
            total += abs(self.columnHeight(c) - self.columnHeight(c + 1))
        return total
    
    def columnHeight(self, column):
        r = 0
        while r < self.rows and self.cells[r, column] == 0:
            r += 1
        return self.rows - r
    
    def addPiece(self, piece):
        for r in range(piece.cells.shape[0]):
            for c in range(piece.cells.shape[1]):
                _r = piece.row + r
                _c = piece.column + c
                if piece.cells[r, c] != 0 and _r >= 0:
                    self.cells[_r, _c] = piece.cells[r, c]

    def valid(self, piece):
        for r in range(piece.cells.shape[0]):
            for c in range(piece.cells.shape[1]):
                _r = piece.row + r
                _c = piece.column + c
                if piece.cells[r, c] != 0:
                    if _r < 0 or _r >= self.rows:
                        return False
                    if _c < 0 or _c >= self.columns:
                        return False
                    if self.cells[_r, _c] != 0:
                        return False
        return True
    
class Piece:
    def __init__(self, cells):
        self.cells = cells
        self.dimension = len(self.cells)
        self.row = 0
        self.column = 0
    
    @staticmethod
    def fromIndex(index):
        if index == 0: # O
            cells = np.array([
                [0x0000AA, 0x0000AA],
                [0x0000AA, 0x0000AA]
            ])
        elif index == 1: # J
            cells = np.array([
                [0xC0C0C0, 0x000000, 0x000000],
                [0xC0C0C0, 0xC0C0C0, 0xC0C0C0],
                [0x000000, 0x000000, 0x000000]
            ])
        elif index == 2: # L
            cells = np.array([
                [0x000000, 0x000000, 0xAA00AA],
                [0xAA00AA, 0xAA00AA, 0xAA00AA],
                [0x000000, 0x000000, 0x000000]
            ])
        elif index == 3: # Z
            cells = np.array([
                [0x00AAAA, 0x00AAAA, 0x000000],
                [0x000000, 0x00AAAA, 0x00AAAA],
                [0x000000, 0x000000, 0x000000]
            ])
        elif index == 4: # S
            cells = np.array([
                [0x000000, 0x00AA00, 0x00AA00],
                [0x00AA00, 0x00AA00, 0x000000],
                [0x000000, 0x000000, 0x000000]
            ])
        elif index == 5: # T
            cells = np.array([
                [0x000000, 0xAA5500, 0x000000],
                [0xAA5500, 0xAA5500, 0xAA5500],
                [0x000000, 0x000000, 0x000000]
            ])
        elif index == 6: # I
            cells = np.array([
                [0x000000, 0x000000, 0x000000, 0x000000],
                [0xAA0000, 0xAA0000, 0xAA0000, 0xAA0000],
                [0x000000, 0x000000, 0x000000, 0x000000],
                [0x000000, 0x000000, 0x000000, 0x000000]
            ])
        piece = Piece(cells)
        piece.row = 0
        piece.column = (10 - piece.dimension) // 2
        return piece
    
    def clone(self):
        piece = Piece(np.copy(self.cells))
        piece.row = self.row
        piece.column = self.column
        return piece
    
    def canMoveLeft(self, grid):
        for r in range(self.cells.shape[0]):
            for c in range(self.cells.shape[1]):
                _r = self.row + r
                _c = self.column + c - 1
                if self.cells[r, c] != 0:
                    if not (_c >= 0 and grid.cells[_r, _c] == 0):
                        return False
        return True
    
    def canMoveRight(self, grid):
        for r in range(self.cells.shape[0]):
            for c in range(self.cells.shape[1]):
                _r = self.row + r
                _c = self.column + c + 1
                if self.cells[r, c] != 0:
                    if not (_c >= 0 and grid.cells[_r, _c] == 0):
                        return False
        return True
    
    def canMoveDown(self, grid):
        for r in range(self.cells.shape[0]):
            for c in range(self.cells.shape[1]):
                _r = self.row + r + 1
                _c = self.column + c
                if self.cells[r, c] != 0 and _r >= 0:
                    if not (_r < grid.rows and grid.cells[_r, _c] == 0):
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
        _cells = np.zeros((self.dimension, self.dimension), dtype=np.int32)
        if self.dimension == 2:
            _cells[0, 0] = self.cells[1, 0]
            _cells[0, 1] = self.cells[0, 0]
            _cells[1, 0] = self.cells[1, 1]
            _cells[1, 1] = self.cells[0, 1]
        elif self.dimension == 3:
            _cells[0, 0] = self.cells[2, 0]
            _cells[0, 1] = self.cells[1, 0]
            _cells[0, 2] = self.cells[0, 0]
            _cells[1, 0] = self.cells[2, 1]
            _cells[1, 1] = self.cells[1, 1]
            _cells[1, 2] = self.cells[0, 1]
            _cells[2, 0] = self.cells[2, 2]
            _cells[2, 1] = self.cells[1, 2]
            _cells[2, 2] = self.cells[0, 2]
        elif self.dimension == 4:
            _cells[0, 0] = self.cells[3, 0]
            _cells[0, 1] = self.cells[2, 0]
            _cells[0, 2] = self.cells[1, 0]
            _cells[0, 3] = self.cells[0, 0]
            _cells[1, 3] = self.cells[0, 1]
            _cells[2, 3] = self.cells[0, 2]
            _cells[3, 3] = self.cells[0, 3]
            _cells[3, 2] = self.cells[1, 3]
            _cells[3, 1] = self.cells[2, 3]
            _cells[3, 0] = self.cells[3, 3]
            _cells[2, 0] = self.cells[3, 2]
            _cells[1, 0] = self.cells[3, 1]
            _cells[1, 1] = self.cells[2, 1]
            _cells[1, 2] = self.cells[1, 1]
            _cells[2, 2] = self.cells[1, 2]
            _cells[2, 1] = self.cells[2, 2]
        self.cells = _cells

    def computeRotateOffset(self, grid):
        _piece = self.clone()
        _piece.rotateCells()
        if grid.valid(_piece):
            return {"rowOffset": _piece.row - self.row, "columnOffset": _piece.column - self.column}
        
        # Kicking
        initialRow = _piece.row
        initialCol = _piece.column

        for i in range(_piece.dimension - 1):
            _piece.column = initialCol + i
            if grid.valid(_piece):
                return {"rowOffset": _piece.row - self.row, "columnOffset": _piece.column - self.column}
            for j in range(_piece.dimension - 1):
                _piece.row = initialRow - j
                if grid.valid(_piece):
                    return {"rowOffset": _piece.row - self.row, "columnOffset": _piece.column - self.column}
            _piece.row = initialRow
        _piece.column = initialCol

        for i in range(_piece.dimension - 1):
            _piece.column = initialCol - i
            if grid.valid(_piece):
                return {"rowOffset": _piece.row - self.row, "columnOffset": _piece.column - self.column}
            for j in range(_piece.dimension - 1):
                _piece.row = initialRow - j
                if grid.valid(_piece):
                    return {"rowOffset": _piece.row - self.row, "columnOffset": _piece.column - self.column}
            _piece.row = initialRow
        _piece.column = initialCol

        return None
    
    def rotate(self, grid):
        offset = self.computeRotateOffset(grid)
        if offset is not None:
            self.rotateCells()
            self.row += offset["rowOffset"]
            self.column += offset["columnOffset"]

class AI:
    def __init__(self, weights=None):
        if weights is None:
            self.heightWeight = 0.510066
            self.linesWeight = 0.760666
            self.holesWeight = 0.35663
            self.bumpinessWeight = 0.184483
        else:
            self.heightWeight = weights["heightWeight"]
            self.linesWeight = weights["linesWeight"]
            self.holesWeight = weights["holesWeight"]
            self.bumpinessWeight = weights["bumpinessWeight"]
    
    def _best(self, grid, workingPieces, workingPieceIndex):
        print("start")
        best = None
        bestScore = None
        workingPiece = workingPieces[workingPieceIndex]

        for rotation in range(4):
            _piece = workingPiece.clone()
            for i in range(rotation):
                _piece.rotate(grid)
            while _piece.moveLeft(grid):
                pass
            while grid.valid(_piece):
                _pieceSet = _piece.clone()
                while _pieceSet.moveDown(grid):
                    pass
                _grid = grid.clone()
                _grid.addPiece(_pieceSet)
                if workingPieceIndex == len(workingPieces) - 1:
                    score = -self.heightWeight * _grid.aggregateHeight() + self.linesWeight * _grid.lines() - self.holesWeight * _grid.holes() - self.bumpinessWeight * _grid.bumpiness()
                    # print(score)
                else:
                    score = self._best(_grid, workingPieces, workingPieceIndex + 1)["score"]
                # if score > bestScore or bestScore is None:
                if bestScore is None or score > bestScore:
                    bestScore = score
                    best = _piece.clone()
                _piece.column += 1
        return {"piece": best, "score": bestScore}

    # write the above function to keep track of moves made
    # L = move left
    # R = move right
    # D = move down
    # T = rotate

    def _best(self, grid, workingPieces, workingPieceIndex, moves):
        best = None
        bestScore = None
        workingPiece = workingPieces[workingPieceIndex]

        for rotation in range(4):
            _piece = workingPiece.clone()
            for i in range(rotation):
                _piece.rotate(grid)
                moves.append("T")
            while _piece.moveLeft(grid):
                moves.append("L")
            while grid.valid(_piece):
                _pieceSet = _piece.clone()
                while _pieceSet.moveDown(grid): pass
                _grid = grid.clone()
                _grid.addPiece(_pieceSet)
                if workingPieceIndex == len(workingPieces) - 1:
                    score = -self.heightWeight * _grid.aggregateHeight() + self.linesWeight * _grid.lines() - self.holesWeight * _grid.holes() - self.bumpinessWeight * _grid.bumpiness()
                else:
                    score = self._best(_grid, workingPieces, workingPieceIndex + 1, moves)["score"]
                # if score > bestScore or bestScore is None:
                if bestScore is None or score > bestScore:
                    bestScore = score
                    best = _piece.clone()
                _piece.column += 1
                moves.append("R")
        return {"piece": best, "score": bestScore}
    
    # def best(self, grid, workingPieces):
    #     res = self._best(grid, workingPieces, 0)
    #     return res["piece"]

    def best(self, grid, workingPieces):
        moves = []
        res = self._best(grid, workingPieces, 0, moves)
        return res["piece"], moves
    