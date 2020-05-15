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
        self.board = np.arange(64, dtype=Figure)
        self.board = self.board.reshape((8, 8))
        self.move_history = []
        self.next_move_available = True
        self.empty_board = empty_board
        self.play = False
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

    def clear_pos(self, pos):
        """
        Clear pos from 0 to 63 on board
        Args:
            pos:

        Returns:

        """
        row = pos // 8
        col = pos % 8
        self.board[row, col] = pos

    def check_moves_left(self):
        pass

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
            self._move_piece(pos_from, pos_to)

    def _move_piece(self, pos_from, pos_to):
        row = pos_from // 8
        col = pos_from % 8
        new_row = pos_to // 8
        new_col = pos_to % 8
        self.board[new_row, new_col] = self.board[row, col]
        self.board[row, col] = pos_from

        if self.board[new_row, new_col].fig_type == 'king':
            if pos_from == 4 and pos_to == 2:
                self.board[0, 3] = self.board[0, 0]
                self.clear_pos(0)
            elif pos_from == 4 and pos_to == 6:
                self.board[0, 5] = self.board[0, 7]
                self.clear_pos(7)
            elif pos_from == 60 and pos_to == 58:
                self.board[7, 3] = self.board[7, 0]
                self.clear_pos(56)
            elif pos_from == 60 and pos_to == 62:
                self.board[7, 5] = self.board[7, 7]
                self.clear_pos(63)

        if self.board[new_row, new_col].fig_type == "pawn" and self.board[new_row, new_col].moved is False \
                and new_col == col and \
                ((row == 1 and new_row == 3)
                 or (row == 6 and new_row == 4)):
            self.board[new_row, new_col].used_turbo = True
        else:
            self.board[new_row, new_col].used_turbo = False

        self.board[new_row, new_col].moved = True
        self.move_history.append({'from': pos_from,
                                  'to': pos_to,
                                  'fig_type': self.board[new_row, new_col].fig_type,
                                  'color': self.board[new_row, new_col].color})

        if (new_row == 7 and self.board[
            new_row, new_col].fig_type == "pawn" and self.board[new_row, new_col].color == "white") \
                or (new_row == 0 and self.board[new_row, new_col].fig_type == "pawn" \
                    and self.board[new_row, new_col].color == "black"):
            if self.play is True:
                while True:
                    upgrade_to = self.select_pawn_upgrade()
                    valid = self.board[new_row, new_col].upgrade_pawn(upgrade_to)
                    if valid:
                        break
            else:
                self.board[new_row, new_col].upgrade_pawn('queen')

    @staticmethod
    def select_pawn_upgrade():
        print("Select figure:\n0 Queen\n1 rook\n2 Knight\n3 Bishop")
        selection = int(input())
        if selection == 0:
            return 'queen'
        elif selection == 1:
            return 'rook'
        elif selection == 2:
            return "knight"
        elif selection == 3:
            return "bishop"
        else:
            return "unkown"

    def check_move_rules(self, pos, target_pos):
        """
        Checks board layout for specific figure.
        Args:
            pos:
            target_pos:

        Returns:
        Boolean: True if valid move
        """
        valid_move = self._check_piece_move_rules(pos, target_pos)
        if not valid_move:
            return False
        if type(self.board[target_pos // 8, target_pos % 8]) is Figure \
                and self.board[target_pos // 8, target_pos % 8].fig_type == 'king':
            return False

        valid_move = self._check_post_move_check(pos, target_pos)
        if not valid_move:
            return False

        return True

    def _check_diagonal_attack_line(self, pos, target_pos):
        row = pos // 8
        col = pos % 8

        target_row = target_pos // 8
        target_col = target_pos % 8

        if row == target_row or col == target_col:
            return False

        elif abs(row - target_row) == abs(col - target_col):
            row_direction = np.sign(target_row - row)
            col_direction = np.sign(target_col - col)
            for step in range(1, abs(row - target_row)):  # indexed 1 -> + 1
                if type(self.board[row + step * row_direction, col + step * col_direction]) is Figure:
                    return False
            return True
        return False

    def _check_king_move(self, pos, target_pos):
        row = pos // 8
        col = pos % 8
        target_row = target_pos // 8
        target_col = target_pos % 8
        king_color = self.board[row, col].color

        if abs(row - target_row) + abs(col - target_col) == 1:
            pass
        elif abs(row - target_row) == 1 and abs(col - target_col) == 1:
            pass
        elif abs(col - target_col) == 2:
            # castling

            if king_color == 'white' and self.is_field_attacked(pos, 'black') or \
                    king_color == 'black' and self.is_field_attacked(pos, 'white'):
                # Can not do castling when being checked
                return False

            if self.board[row, col].moved \
                    or king_color == 'white' and pos != 4 \
                    or king_color == 'black' and pos != 60:
                # King has moved
                return False

            if 'white' == self.board[row, col].color and target_pos == 2:
                if not (type(self.board[0, 0]) is Figure and not self.board[0, 0].moved
                        and self.board[0, 0].color == 'white' and self.board[0, 0].fig_type == 'rook'):
                    return False
                if type(self.board[0, 1]) is Figure \
                        or type(self.board[0, 2]) is Figure \
                        or type(self.board[0, 3]) is Figure \
                        or self.is_field_attacked(3, 'black'):
                    return False
            elif 'white' == self.board[row, col].color and target_pos == 6:
                if not (type(self.board[0, 7]) is Figure and not self.board[0, 7].moved
                        and self.board[0, 7].color == 'white' and self.board[0, 7].fig_type == 'rook'):
                    return False
                if type(self.board[0, 5]) is Figure \
                        or type(self.board[0, 6]) is Figure \
                        or self.is_field_attacked(5, 'black') \
                        or self.is_field_attacked(6, 'black'):
                    return False
            elif 'black' == self.board[row, col].color and target_pos == 58:
                if not (type(self.board[7, 0]) is Figure and not self.board[7, 0].moved
                        and self.board[7, 0].color == 'black' and self.board[7, 0].fig_type == 'rook'):
                    return False
                if type(self.board[7, 1]) is Figure \
                        or type(self.board[7, 2]) is Figure \
                        or type(self.board[7, 3]) is Figure \
                        or self.is_field_attacked(59, 'white'):
                    return False
            elif 'black' == self.board[row, col].color and target_pos == 62:
                if not (type(self.board[7, 7]) is Figure and not self.board[7, 7].moved
                        and self.board[7, 7].color == 'black' and self.board[7, 7].fig_type == 'rook'):
                    return False
                if type(self.board[7, 5]) is Figure \
                        or type(self.board[7, 6]) is Figure \
                        or self.is_field_attacked(62, 'white'):
                    return False
            else:
                return False
            # if 'white' == self.board[row, col].color and target_pos == 2:
            #     if not (type(self.board[0, 0]) is Figure and not self.board[0, 0].moved):
            #         return None
            #     if self.is_field_attacked(3, 'black'):
            #         return False
            #     return True
            # if 'white' == self.board[row, col].color and target_pos == 2:
            #     if not (type(self.board[0, 0]) is Figure and not self.board[0, 0].moved):
            #         return None
            #     if self.is_field_attacked(3, 'black'):
            #         return False
            #     return True
        else:
            return False

        if 'white' == self.board[row, col].color:
            valid_distance = self.check_enemy_king_distance(target_pos, 'black')
        else:
            valid_distance = self.check_enemy_king_distance(target_pos, 'white')
        return valid_distance

    def is_field_attacked(self, field, attacker_color):
        for row, row_element in enumerate(self.board):
            for col, figure in enumerate(row_element):
                if type(figure) is Figure and figure.color == attacker_color \
                        and figure.fig_type != 'king':
                    if self._check_piece_move_rules(row * 8 + col, field):
                        return True
                else:
                    continue
        return False

    def check_enemy_king_distance(self, position, enemy_king_color):
        row = position // 8
        col = position % 8

        for row_x in range(3):
            for col_x in range(3):
                checking_row = row - 1 + row_x
                checking_col = col - 1 + col_x
                if checking_row < 0 or checking_row > 7 \
                        or checking_col < 0 or checking_col > 7:
                    continue
                if type(self.board[checking_row, checking_col]) is Figure \
                        and self.board[checking_row, checking_col].fig_type == 'king' \
                        and self.board[checking_row, checking_col].color == enemy_king_color:
                    return False
        return True

    def _check_not_diagonal_attack_line(self, pos, target_pos):
        row = pos // 8
        col = pos % 8
        target_row = target_pos // 8
        target_col = target_pos % 8

        row_direction = np.sign(target_row - row)
        col_direction = np.sign(target_col - col)
        if row_direction and col_direction:
            return False

        if row_direction:
            for step in range(1, abs(row - target_row)):  # indexed 1 -> + 1
                if type(self.board[row + step * row_direction, col]) is Figure:
                    return False
        elif col_direction:
            for step in range(1, abs(col - target_col)):  # indexed 1 -> + 1
                if type(self.board[row, col + step * col_direction]) is Figure:
                    return False
        return True

    def _check_post_move_check(self, pos, target_pos):
        """
        Returns information if move is valid
        Args:
            pos:
            target_pos:

        Returns:

        """
        new_game = ChessGame(empty_board=True)
        new_game.board = self.board.copy()
        current_color = new_game.board[pos // 8, pos % 8].color
        new_game.board[target_pos // 8, target_pos % 8] = new_game.board[pos // 8, pos % 8]
        new_game.clear_pos(pos)

        king_pos = new_game.find_king_position(current_color)

        if king_pos is None:
            return True

        if current_color == 'white':
            post_move_check = new_game.is_field_attacked(king_pos, 'black')
        else:
            post_move_check = new_game.is_field_attacked(king_pos, 'white')
        return not post_move_check

    def find_king_position(self, color):
        """
        Returns first king position
        Args:
            color:

        Returns:

        """
        for row, row_element in enumerate(self.board):
            for col, figure in enumerate(row_element):
                if type(figure) is Figure and figure.color == color and figure.fig_type == 'king':
                    return row * 8 + col
        return None

    def _check_piece_move_rules(self, pos, target_pos):
        pos = int(pos)
        target_pos = int(target_pos)

        # Checking positions ranges
        if pos == target_pos:
            return False
        elif pos > 63 or pos < 0 or target_pos > 63 or target_pos < 0:
            return False

        row = pos // 8
        col = pos % 8
        target_row = target_pos // 8
        target_col = target_pos % 8

        try:
            attacker_color = self.board[row, col].color
            attacker_type = self.board[row, col].fig_type
        except AttributeError:
            return False

        try:
            defender_color = self.board[target_row, target_col].color
            defender_type = self.board[target_row, target_col].fig_type
        except AttributeError:
            defender_color = None
            defender_type = None

        if attacker_color == defender_color:
            return False

        if attacker_type == 'pawn':
            if attacker_color == 'white' and target_pos < pos \
                    or attacker_color == 'black' and target_pos > pos:
                return False

            if defender_type is None and abs(target_pos - pos) == 8:
                # Move forward to empty space
                return True
            elif defender_type and abs(target_pos - pos) in [7, 9] \
                    and attacker_color != self.board[target_row, target_col].color:
                # Pawn Attack
                return True
            elif type(self.board[target_row, target_col]) is int and abs(target_pos - pos) in [7, 9]:
                # Counter Turbo Move
                if attacker_color == 'white' and type(self.board[target_row - 1, target_col]) is Figure:
                    if self.board[target_row - 1, target_col].fig_type == 'pawn' \
                            and self.board[target_row - 1, target_col].used_turbo is True \
                            and self.move_history[-1].get('to', -1) == (target_row - 1) * 8 + target_col:
                        return True

                elif attacker_color == 'black' and type(self.board[target_row + 1, target_col]) is Figure:
                    if self.board[target_row + 1, target_col].fig_type == 'pawn' \
                            and self.board[target_row + 1, target_col].used_turbo is True \
                            and self.move_history[-1].get('to', -1) == (target_row + 1) * 8 + target_col:
                        return True

            elif abs(target_pos - pos) == 16:
                # Make Turbo Move
                if type(self.board[target_row, target_col]) is not int:
                    return False
                if attacker_color == 'black' \
                        and pos in range(48, 56) and type(self.board[target_row + 1, target_col]) is int:
                    return True

                elif attacker_color == 'white' \
                        and pos in range(8, 16) and type(self.board[target_row - 1, target_col]) is int:
                    return True

        elif attacker_type == 'rook':
            valid = self._check_not_diagonal_attack_line(pos, target_pos)
            return valid

        elif attacker_type == 'knight':
            if abs(target_row - row) == 2 and abs(target_col - col) == 1 \
                    or abs(target_row - row) == 1 and abs(target_col - col) == 2:
                return True
        elif attacker_type == 'bishop':
            valid = self._check_diagonal_attack_line(pos, target_pos)
            return valid

        elif attacker_type == 'queen':

            valid = self._check_diagonal_attack_line(pos, target_pos) \
                    or self._check_not_diagonal_attack_line(pos, target_pos)
            return valid

        elif attacker_type == 'king':
            valid = self._check_king_move(pos, target_pos)
            return valid
        else:
            raise ValueError(f"Figure type unknown: {attacker_type}")

        return False


class Figure:
    def __init__(self, initial_position, color=None, fig_type=None):
        if type(initial_position) != int:
            raise ValueError("Position is not int")

        self.initial_position = initial_position
        self.moved = False
        self.used_turbo = False

        if color is None and fig_type is None:
            self._place_default()
        else:
            if color is None or type(color) != str:
                raise ValueError("Color is not valid")
            self.color = color

            if fig_type != "king" and fig_type != "queen" and fig_type != "rook" and \
                    fig_type != "bishop" and fig_type != "knight" and fig_type != "pawn":
                raise ValueError(f"Figure type invalid: '{fig_type}'")
            self.fig_type = fig_type

    def upgrade_pawn(self, target_type):
        if self.fig_type == 'pawn' and target_type in ['queen', 'rook', 'bishop', 'knight']:
            self.fig_type = target_type
            return True
        return False

    def _place_default(self):
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
