import numpy as np

MOVES = {
        56: 'h1', 57: 'h2', 58: 'h3', 59: 'h4', 60: 'h5', 61: 'h6', 62: 'h7', 63: 'h8',
        48: 'g1', 49: 'g2', 50: 'g3', 51: 'g4', 52: 'g5', 53: 'g6', 54: 'g7', 55: 'g8',
        40: 'f1', 41: 'f2', 42: 'f3', 43: 'f4', 44: 'f5', 45: 'f6', 46: 'f7', 47: 'f8',
        32: 'e1', 33: 'e2', 34: 'e3', 35: 'e4', 36: 'e5', 37: 'e6', 38: 'e7', 39: 'e8',
        24: 'd1', 25: 'd2', 26: 'd3', 27: 'd4', 28: 'd5', 29: 'd6', 30: 'd7', 31: 'd8',
        16: 'c1', 17: 'c2', 18: 'c3', 19: 'c4', 20: 'c5', 21: 'c6', 22: 'c7', 23: 'c8',
        8: 'b1', 9: 'b2', 10: 'b3', 11: 'b4', 12: 'b5', 13: 'b6', 14: 'b7', 15: 'b8',
        0: 'a1', 1: 'a2', 2: 'a3', 3: 'a4', 4: 'a5', 5: 'a6', 6: 'a7', 7: 'a8'
}


class ChessGame:
    def __init__(self):
        self.empty_field = 0
        self.board = np.zeros((8, 8), dtype=Figure)
        self.move_history = []
        self.next_move_available = True
        self.reset()

    def reset(self):
        self.board = np.zeros((8, 8), dtype=Figure)
        self.board[:] = self.empty_field
        self.move_history = []
        self.next_move_available = True
        for row in [0, 1, 6, 7]:
            for column in range(8):
                self.board[row, column] = Figure(row * 8 + column)

        return self.getstate()

    def getstate(self):
        """
        Whites are at index <0, 15>
        Black are at index <48, 63>
        to get correctly rotated board, use pretty_state
        :return:
        numpy array, shape=(8,8), dtype=Figure
        """
        return self.board.copy()

    def pretty_state(self):
        """
        Whites are at index <0, 15>
        Black are at index <48, 63>
        Returns array rotated for good look in console
        :return:
        numpy array, shape=(8,8), dtype=Figure
        """
        out = self.board.copy()
        out = np.flip(out)
        out = np.fliplr(out)
        return out

    def make_move(self, pos_from, pos_to):
        valid = self.check_move_rules(pos_from, pos_to)
        if valid:
            row = pos_from // 8
            col = pos_from % 8
            new_row = pos_to // 8
            new_col = pos_to % 8
            self.board[new_row, new_col] = self.board[row, col]
            self.board[row, col] = self.empty_field

    def check_move_rules(self, pos_from, pos_to):
        return True


class Figure:
    def __init__(self, position, color=None, fig_type=None):
        if type(position) != int:
            raise ValueError("Position is not int")

        self.position = position

        if color is None and fig_type is None:
            self._set_to_default_figure()
        else:
            if color is None or type(color) != str:
                raise ValueError("Color is not valid")
            self.color = color

            if fig_type != "king" and fig_type != "queen" and fig_type != "rook" and \
                    fig_type != "bishop" and fig_type != "knight" and fig_type != "pawn":
                raise ValueError("Figure type invalid")
            self.fig_type = fig_type

    def _set_to_default_figure(self):
        if self.position <= 16:
            self.color = 'white'
        elif self.position >= 47:
            self.color = 'black'
        else:
            raise ValueError(f"Can not define default color for position {self.position}")

        if 8 <= self.position <= 15 or 48 <= self.position <= 55:
            self.fig_type = 'pawn'

        elif self.position in [0, 7, 56, 63]:
            self.fig_type = 'rook'
        elif self.position in [1, 6, 57, 62]:
            self.fig_type = 'knight'
        elif self.position in [2, 5, 58, 61]:
            self.fig_type = 'bishop'
        elif self.position in [3, 59]:
            self.fig_type = 'queen'
        elif self.position in [4, 60]:
            self.fig_type = 'king'
        else:
            raise ValueError("Position uknown, can not set default figure")

    def __repr__(self):
        _fig_letter = None
        if self.fig_type == 'queen':
            if self.color == 'white':
                _fig_letter = '♕'
            else:
                _fig_letter = '♛'

        elif self.fig_type == 'king':

            if self.color == 'white':
                _fig_letter = '♔'
            else:
                _fig_letter = '♚'

        elif self.fig_type == 'rook':
            if self.color == 'white':
                _fig_letter = '♖'
            else:
                _fig_letter = '♜'

        elif self.fig_type == 'bishop':
            if self.color == 'white':
                _fig_letter = '♗'
            else:
                _fig_letter = '♝'

        elif self.fig_type == 'knight':
            if self.color == 'white':
                _fig_letter = '♘'
            else:
                _fig_letter = '♞'
        elif self.fig_type == 'pawn':
            if self.color == 'white':
                _fig_letter = '♙'
            else:
                _fig_letter = '♟'
        else:
            raise ValueError(f"Repr failed on this fig_type: '{self.fig_type}', fig_color: '{self.color}'")
        return _fig_letter


if __name__ == "__main__":
    game = ChessGame()
    print(game.pretty_state())
    game.make_move(0, 16)
    print(game.pretty_state())
