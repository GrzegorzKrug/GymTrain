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
    print_initial_board(game)

    for field, step_size, valid in zip(black_fields, [8, 16] * 4,
                                       [False, False, True, False, True, False, True, False]):
        result = game.check_move_rules(field, field - step_size)
        assert result is valid
    assert False


def test_pawn_attack_white():
    game = ChessGame(empty_board=True)
    fields = [16, 9, 34, 14, 36]
    for field in fields:
        game.add_figure(field, 'pawn', 'white')
    game.add_figure(25, 'rook', 'black')
    game.add_figure(18, 'rook', 'black')
    game.add_figure(41, 'queen', 'black')
    game.add_figure(21, 'bishop', 'black')
    game.add_figure(45, 'king', 'black')
    print_initial_board(game)

    assert game.check_move_rules(16, 25)
    assert game.check_move_rules(9, 18)
    assert game.check_move_rules(14, 21)
    assert game.check_move_rules(34, 41)
    assert not game.check_move_rules(36, 45)


def test_pawn_attack_black():
    game = ChessGame(empty_board=True)
    fields = [16, 9, 34, 14, 36]
    for field in fields:
        game.add_figure(field, 'pawn', 'white')
    game.add_figure(25, 'rook', 'black')
    game.add_figure(18, 'rook', 'black')
    game.add_figure(41, 'queen', 'black')
    game.add_figure(21, 'bishop', 'black')
    game.add_figure(45, 'king', 'black')
    game.flip()
    print_initial_board(game)

    assert game.check_move_rules(63 - 16, 63 - 25)
    assert game.check_move_rules(63 - 9, 63 - 18)
    assert game.check_move_rules(63 - 14, 63 - 21)
    assert game.check_move_rules(63 - 34, 63 - 41)
    assert not game.check_move_rules(63 - 36, 63 - 45)


def test_turbo_hit_white():
    game = ChessGame()
    game.add_figure(32, 'pawn', 'white')
    game.add_figure(35, 'pawn', 'white')
    game.add_figure(38, 'pawn', 'white')
    print_initial_board(game)

    game.make_move(49, 33)
    print(f"Current state:\n{game.pretty_state()}")
    assert game.check_move_rules(32, 41)
    assert not game.check_move_rules(35, 42)

    game.make_move(50, 34)
    print(f"Current state:\n{game.pretty_state()}")
    assert not game.check_move_rules(32, 41)
    assert game.check_move_rules(35, 42)

    game.make_move(53, 45)
    assert game.check_move_rules(38, 45)
    game.make_move(45, 37)
    assert not game.check_move_rules(38, 45)


def test_turbo_hit_black():
    game = ChessGame()
    game.add_figure(32, 'pawn', 'white')
    game.add_figure(35, 'pawn', 'white')
    game.add_figure(38, 'pawn', 'white')
    game.flip()
    print_initial_board(game)

    game.make_move(63 - 49, 63 - 33)
    print(f"Current state:\n{game.pretty_state()}")
    assert game.check_move_rules(63 - 32, 63 - 41)
    assert not game.check_move_rules(63 - 35, 63 - 42)

    game.make_move(63 - 50, 63 - 34)
    print(f"Current state:\n{game.pretty_state()}")
    assert not game.check_move_rules(63 - 32, 63 - 41)
    assert game.check_move_rules(63 - 35, 63 - 42)

    game.make_move(63 - 53, 63 - 45)
    assert game.check_move_rules(63 - 38, 63 - 45)
    game.make_move(63 - 45, 63 - 37)
    assert not game.check_move_rules(63 - 38, 63 - 45)


def test_move_back():
    game = ChessGame(empty_board=True)
    game.add_figure(40, 'pawn', 'white')
    game.add_figure(26, 'pawn', 'black')
    print_initial_board(game)

    assert not game.check_move_rules(40 - 8)
    assert not game.check_move_rules(26 + 8)


def test_pawn_upgrade():
    game = ChessGame(empty_board=True)
    game.add_figure(48, 'pawn', 'white')
    game.add_figure(8, 'pawn', 'black')
    print_initial_board(game)

    game.make_move(48, 56)
    assert game.board[7, 0].fig_type in ['rook', 'knight', 'queen', 'bishop']
    game.make_move(8, 0)
    assert game.board[0, 0].fig_type in ['rook', 'knight', 'queen', 'bishop']


def print_initial_board(game):
    print(f"Initial board:\nBlack\n{game.pretty_state()}\nWhite")


def print_initial_board_flipped(game):
    print(f"Initial board FLIPPED:\nWhite\n{game.getstate()}\nBlack")


def test_intended_fail():
    game = ChessGame(empty_board=True)
    print_initial_board(game)
    raise RuntimeError("Intended stop")
