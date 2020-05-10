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
    def __init__(self, empty_board=False):
        self.empty_field = 0
        self.board = np.arange(64, dtype=Figure)
        self.board = self.board.reshape((8, 8))
        self.move_history = []
        self.next_move_available = True
        self.empty_board = empty_board
        self.reset()

    def reset(self):
        """
        Setups all variables defined in __init__
        Returns:
        Returns state array.
        """
        self.board = np.arange(64, dtype=Figure)
        self.board = self.board.reshape((8, 8))
        # self.board[:] = self.empty_field
        self.move_history = []

        if self.empty_board:
            self.next_move_available = False
        else:
            self.next_move_available = True
            for row in [0, 1, 6, 7]:
                for column in range(8):
                    self.board[row, column] = Figure(row * 8 + column)
        return self.getstate()

    def add_figure(self, pos, fig_type, color):
        row = pos // 8
        col = pos % 8
        self.board[row, col] = Figure(initial_position=pos, color=color, fig_type=fig_type)

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
        Returns pretty array for console print.
            Whites are at index <0, 15>
            Black are at index <48, 63>
        :return:
        numpy array, shape=(8,8), dtype=Figure
        """
        out = self.board.copy()
        out = np.flipud(out)
        table = ''
        for num, row in enumerate(out):
            for element in row:
                if type(element) is int:
                    table += f"{element}".rjust(2, '0').ljust(3)
                else:
                    table += f"{element}".ljust(3)
            if num < 7:
                table += '\n'
        return table

    def flip(self):
        self.flip_state()

    def flip_state(self):
        """
        Set layout to match exact situation with reversed colors
        Returns:

        """
        self.board = np.flip(self.board)
        for row in self.board:
            for figure in row:
                if type(figure) is not int:
                    figure.color = 'black' if figure.color == 'white' else 'white'

    def make_move(self, pos_from, pos_to):
        valid = self.check_move_rules(pos_from, pos_to)
        if valid:
            row = pos_from // 8
            col = pos_from % 8
            new_row = pos_to // 8
            new_col = pos_to % 8
            self.board[new_row, new_col] = self.board[row, col]
            self.board[row, col] = self.empty_field

    def check_move_rules(self, pos, target_pos):
        """
        Checks board layout for specific figure.
        Args:
            pos:
            target_pos:

        Returns:
        Boolean: True if valid move
        """
        row = pos // 8
        col = pos % 8
        target_row = target_pos // 8
        target_col = target_pos % 8

        if type(self.board[target_row, target_col]) != int and \
                (self.board[row, col].color == self.board[target_row, target_col].color
                 or self.board[target_row, target_col].fig_type == 'king'):
            return False

        figure = self.board[row, col]
        fig_type = figure.fig_type

        if fig_type == 'pawn':
            if figure.color == 'white':
                if target_pos - pos in [7, 8, 9]:
                    return True

                elif target_pos - pos == 16:
                    if self.board[row + 1, col] == 0 and self.figure:
                        return True
                    else:
                        return False
            elif figure.color == 'black':
                pass
        elif fig_type == 'rook':
            pass
        elif fig_type == 'knight':
            pass
        elif fig_type == 'bishop':
            pass
        elif fig_type == 'queen':
            pass
        elif fig_type == 'king':
            pass
        else:
            raise ValueError(f"Figure type unknown: {fig_type}")


class Figure:
    def __init__(self, initial_position, color=None, fig_type=None):
        if type(initial_position) != int:
            raise ValueError("Position is not int")

        self.initial_position = initial_position
        self.moved = False

        if color is None and fig_type is None:
            self._set_to_default_figure()
        else:
            if color is None or type(color) != str:
                raise ValueError("Color is not valid")
            self.color = color

            if fig_type != "king" and fig_type != "queen" and fig_type != "rook" and \
                    fig_type != "bishop" and fig_type != "knight" and fig_type != "pawn":
                raise ValueError(f"Figure type invalid: '{fig_type}'")
            self.fig_type = fig_type

    def _set_to_default_figure(self):
        if self.initial_position <= 16:
            self.color = 'white'
        elif self.initial_position >= 47:
            self.color = 'black'
        else:
            raise ValueError(f"Can not define default color for initial_position {self.initial_position}")

        if 8 <= self.initial_position <= 15 or 48 <= self.initial_position <= 55:
            self.fig_type = 'pawn'

        elif self.initial_position in [0, 7, 56, 63]:
            self.fig_type = 'rook'
        elif self.initial_position in [1, 6, 57, 62]:
            self.fig_type = 'knight'
        elif self.initial_position in [2, 5, 58, 61]:
            self.fig_type = 'bishop'
        elif self.initial_position in [3, 59]:
            self.fig_type = 'queen'
        elif self.initial_position in [4, 60]:
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
