from chess_game import ChessGame
import numpy as np


def test_pawn_initial_straight_moves():
    game = ChessGame()
    print_initial_board(game)
    for step_size, valid in zip([8, 16, 24], [True, True, False]):
        for x in range(8, 16):
            result = game.check_move_rules(x, x + step_size)
            assert result is valid
        for x in range(48, 56):
            result = game.check_move_rules(x, x - step_size)
            assert result is valid


def test_pawn_straight_moves_in_middle_white():
    game = ChessGame(empty_board=True)
    fields = [16, 25, 34, 43, 52]
    for field in fields:
        game.add_figure(field, "pawn", 'white')
    print_initial_board(game)

    for step_size, valid in zip([8, 16, 24], [True, False, False]):
        for field in fields:
            result = game.check_move_rules(field, field + step_size)
            assert result is valid


def test_pawn_straight_moves_in_middle_black():
    game = ChessGame(empty_board=True)
    fields = [47, 38, 29, 20, 11]
    for field in fields:
        game.add_figure(field, "pawn", 'black')
    print_initial_board(game)

    for step_size, valid in zip([8, 16, 24], [True, False, False]):
        for field in fields:
            result = game.check_move_rules(field, field - step_size)
            assert result is valid


def test_pawn_straight_moves_blocked_white():
    game = ChessGame(empty_board=True)
    white_fields = [8, 9, 10, 11, 20, 21, 38, 47]
    black_fields = [16, 17, 26, 27, 36, 37, 54, 55]
    for white, black in zip(white_fields, black_fields):
        game.add_figure(white, "pawn", "white")
        game.add_figure(black, "pawn", "black")
    print_initial_board(game)

    for field, step_size, valid in zip(white_fields, [8, 16] * 4,
                                       [False, False, True, False, True, False, True, False]):
        result = game.check_move_rules(field, field + step_size)
        assert result is valid
    assert False


def test_pawn_straight_moves_blocked_black():
    game = ChessGame(empty_board=True)
    black_fields = [8, 9, 10, 11, 20, 21, 38, 47]
    white_fields = [16, 17, 26, 27, 36, 37, 54, 55]
    for white, black in zip(white_fields, black_fields):
        game.add_figure(63 - white, "pawn", "white")
        game.add_figure(63 - black, "pawn", "black")
    print_initial_board_flipped(game)

    for field, step_size, valid in zip(black_fields, [8, 16] * 4,
                                       [False, False, True, False, True, False, True, False]):
        result = game.check_move_rules(field, field - step_size)
        assert result is valid
    assert False


def print_initial_board(game):
    print(f"Initial board:\nBlack\n{game.pretty_state()}\nWhite")


def print_initial_board_flipped(game):
    print(f"Initial board:\nWhite\n{game.getstate()}\nBlack")


# def test_intended_fail():
#     indexes = np.arange(64)
#     indexes = indexes.reshape(8, 8)
#     indexes = np.flipud(indexes)
#     print(indexes)
#     raise RuntimeError("Intended stop")
