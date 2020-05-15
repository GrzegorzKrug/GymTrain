from chess_game import ChessGame
import numpy as np


def test_pawn_initial_straight_moves():
    game = ChessGame()
    print_board_state(game)
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
    print_board_state(game)

    for step_size, valid in zip([8, 16, 24], [True, False, False]):
        for field in fields:
            result = game.check_move_rules(field, field + step_size)
            assert result is valid, f"From {field} to {field + step_size}"


def test_pawn_straight_moves_in_middle_black():
    game = ChessGame(empty_board=True)
    fields = [47, 38, 29, 20, 11]
    for field in fields:
        game.add_figure(field, "pawn", 'black')
    print_board_state(game)

    for step_size, valid in zip([8, 16, 24], [True, False, False]):
        for field in fields:
            result = game.check_move_rules(field, field - step_size)
            assert result is valid, f"From {field} to {field - step_size}"


def test_pawn_straight_moves_blocked_white():
    game = ChessGame(empty_board=True)
    white_fields = [8, 9, 10, 11, 20, 21, 38, 47]
    black_fields = [16, 17, 26, 27, 36, 37, 54, 55]
    for white, black in zip(white_fields, black_fields):
        game.add_figure(white, "pawn", "white")
        game.add_figure(black, "pawn", "black")
    print_board_state(game)

    for field, step_size, valid in zip(white_fields, [8, 16] * 4,
                                       [False, False, True, False, True, False, True, False]):
        result = game.check_move_rules(field, field + step_size)
        assert result is valid, f"From {field} to {field + step_size}"


def test_pawn_straight_moves_blocked_black():
    game = ChessGame(empty_board=True)
    black_fields = [8, 9, 10, 11, 20, 21, 38, 47]
    white_fields = [16, 17, 26, 27, 36, 37, 54, 55]
    for white, black in zip(white_fields, black_fields):
        game.add_figure(63 - white, "pawn", "white")
        game.add_figure(63 - black, "pawn", "black")
    print_board_state(game)

    for field, step_size, valid in zip(black_fields, [8, 16] * 4,
                                       [False, False, True, False, True, False, True, False]):
        field = 63 - field
        result = game.check_move_rules(field, field - step_size)
        assert result is valid, f"From {field} to {field - step_size}"


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
    print_board_state(game)

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
    print_board_state(game)

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
    print_board_state(game)

    game.make_move(49, 33)
    print_board_state(game)
    assert game.board[33 // 8, 33 % 8].used_turbo
    assert game.check_move_rules(32, 41), f"Move 32 -> 41"
    assert not game.check_move_rules(35, 42), f"Move 35 -> 42"

    game.make_move(50, 34)
    print_board_state(game)
    assert not game.check_move_rules(32, 41), f"Move 32 -> 41"
    assert game.check_move_rules(35, 42), f"Move 35 -> 42"

    game.make_move(53, 45)
    print_board_state(game)
    assert game.check_move_rules(38, 45), f"Move 38 -> 45"

    game.make_move(45, 37)
    print_board_state(game)
    assert not game.check_move_rules(38, 45), f"Move 38 -> 45"


def test_turbo_hit_black():
    game = ChessGame()
    game.add_figure(32, 'pawn', 'white')
    game.add_figure(35, 'pawn', 'white')
    game.add_figure(38, 'pawn', 'white')
    game.flip()

    game.make_move(63 - 49, 63 - 33)
    print_board_state(game)
    assert game.check_move_rules(63 - 32, 63 - 41), f"{63 - 32} to {63 - 41}"
    assert not game.check_move_rules(63 - 35, 63 - 42), f"{63 - 35} to {63 - 42}"

    game.make_move(63 - 50, 63 - 34)
    print_board_state(game)
    assert not game.check_move_rules(63 - 32, 63 - 41), f"{63 - 32} to {63 - 41}"
    assert game.check_move_rules(63 - 35, 63 - 42), f"{63 - 35} to {63 - 42}"

    game.make_move(63 - 53, 63 - 45)
    print_board_state(game)
    assert game.check_move_rules(63 - 38, 63 - 45), f"{63 - 38} to {63 - 45}"

    game.make_move(63 - 45, 63 - 37)
    print_board_state(game)
    assert not game.check_move_rules(63 - 38, 63 - 45), f"{63 - 38} to {63 - 45}"


def test_move_back():
    game = ChessGame(empty_board=True)
    game.add_figure(40, 'pawn', 'white')
    game.add_figure(26, 'pawn', 'black')
    print_board_state(game)

    assert not game.check_move_rules(40, 40 - 8)
    assert not game.check_move_rules(26, 26 + 8)


def test_pawn_upgrade():
    game = ChessGame(empty_board=True)
    game.add_figure(48, 'pawn', 'white')
    game.add_figure(8, 'pawn', 'black')
    print_board_state(game)

    game.make_move(48, 56)
    assert game.board[7, 0].fig_type in ['rook', 'knight', 'queen', 'bishop']
    assert game.board[7, 0].color == 'white'
    game.make_move(8, 0)
    assert game.board[0, 0].fig_type in ['rook', 'knight', 'queen', 'bishop']
    assert game.board[0, 0].color == 'black'


def test_knight_move():
    game = ChessGame(empty_board=True)
    game.add_figure(9, 'knight', 'white')
    game.add_figure(19, 'knight', 'white')
    game.add_figure(25, 'knight', 'black')
    game.add_figure(35, 'knight', 'black')

    print_board_state(game)

    # first white knight
    valid_fields = [3, 24, 26]
    for x in range(64):
        if x in valid_fields:
            assert game.check_move_rules(9, x)
        else:
            assert not game.check_move_rules(9, x)

    # second white knight
    valid_fields = [25, 34, 36, 29, 13, 4, 2]
    for x in range(64):
        if x in valid_fields:
            assert game.check_move_rules(19, x)
        else:
            assert not game.check_move_rules(19, x)

    # first black knight
    valid_fields = [40, 42, 19, 8, 10]
    for x in range(64):
        if x in valid_fields:
            assert game.check_move_rules(25, x)
        else:
            assert not game.check_move_rules(25, x)

    # second black knight
    valid_fields = [41, 50, 52, 45, 29, 20, 18]
    for x in range(64):
        if x in valid_fields:
            assert game.check_move_rules(35, x)
        else:
            assert not game.check_move_rules(35, x)


def test_bishop_move():
    game = ChessGame(empty_board=True)

    game.add_figure(33, 'bishop', 'white')
    game.add_figure(36, 'bishop', 'white')
    game.add_figure(9, 'bishop', 'black')
    game.add_figure(19, 'bishop', 'black')

    # baits
    game.add_figure(40, 'bishop', 'white')
    game.add_figure(54, 'bishop', 'white')
    game.add_figure(12, 'bishop', 'black')

    print_board_state(game)

    # first black bishop
    good_fields = [42, 51, 60, 26, 19, 24]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(33, field), f"From 33 to {field}"
        else:
            assert not game.check_move_rules(33, field), f"From 33 to {field}"

    # second black bishop
    good_fields = [27, 18, 9, 29, 22, 15, 45, 43, 50, 57]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(36, field), f"From 36 to {field}"
        else:
            assert not game.check_move_rules(36, field), f"From 36 to {field}"

    # first white bishop
    good_fields = [16, 0, 2, 18, 27, 36]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(9, field), f"From 9 to {field}"
        else:
            assert not game.check_move_rules(9, field), f"From 9 to {field}"

    # second white bishop
    good_fields = [1, 10, 26, 33, 28, 37, 46, 55]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(19, field), f"From 19 to {field}"
        else:
            assert not game.check_move_rules(19, field), f"From 19 to {field}"


def test_rook_move():
    game = ChessGame(empty_board=True)
    game.add_figure(9, 'rook', 'white')
    game.add_figure(22, 'rook', 'white')
    game.add_figure(28, 'rook', 'white')
    game.add_figure(49, 'rook', 'black')
    game.add_figure(52, 'rook', 'black')
    game.add_figure(45, 'rook', 'black')

    # Bait
    game.add_figure(12, 'rook', 'white')
    game.add_figure(60, 'rook', 'black')
    print_board_state(game)

    # first white rook
    good_fields = [8, 10, 11, 1, 17, 25, 33, 41, 49]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(9, field), f"Move 9 to {field}"
        else:
            assert not game.check_move_rules(9, field), f"Move 9 to {field}"

    # second white rook
    good_fields = [10, 11, 4, 13, 14, 15, 20]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(12, field), f"Move 12 to {field}"
        else:
            assert not game.check_move_rules(12, field), f"Move 12 to {field}"

    good_fields = [24, 25, 26, 27, 29, 30, 31, 20, 36, 44, 52]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(28, field), f"Move 28 to {field}"
        else:
            assert not game.check_move_rules(28, field), f"Move 28 to {field}"

    good_fields = [16, 17, 18, 19, 20, 21, 23, 14, 6, 30, 38, 46, 54, 62]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(22, field), f"Move 22 to {field}"
        else:
            assert not game.check_move_rules(22, field), f"Move 22 to {field}"

    good_fields = [48, 57, 41, 33, 25, 17, 9, 50, 51]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(49, field), f"Move 49 to {field}"
        else:
            assert not game.check_move_rules(49, field), f"Move 49 to {field}"

    good_fields = [50, 51, 53, 54, 55, 44, 36, 28]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(52, field), f"Move 52 to {field}"
        else:
            assert not game.check_move_rules(52, field), f"Move 52 to {field}"

    good_fields = [40, 41, 42, 43, 44, 46, 47, 53, 61, 37, 29, 21, 13, 5]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(45, field), f"Move 45 to {field}"
        else:
            assert not game.check_move_rules(45, field), f"Move 45 to {field}"


def test_queen_move():
    game = ChessGame(empty_board=True)

    game.add_figure(9, 'queen', 'white')
    game.add_figure(28, 'queen', 'white')
    game.add_figure(33, 'queen', 'black')
    game.add_figure(52, 'queen', 'black')

    # baits
    game.add_figure(0, 'queen', 'white')
    game.add_figure(12, 'queen', 'white')
    game.add_figure(49, 'queen', 'black')
    game.add_figure(54, 'queen', 'black')
    game.add_figure(63, 'queen', 'black')
    print_board_state(game)

    good_fields = [8, 10, 11, 1, 2, 18, 27, 36, 45, 54, 17, 25, 33, 16]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(9, field), f'From 9 to {field}'
        else:
            assert not game.check_move_rules(9, field), f'From 9 to {field}'

    good_fields = [24, 25, 26, 27, 19, 10, 1, 20, 21, 14, 7, 29, 30, 31,
                   37, 46, 55, 36, 44, 52, 35, 42, 49]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(28, field), f'From 28 to {field}'
        else:
            assert not game.check_move_rules(28, field), f'From 28 to {field}'

    good_fields = [32, 40, 41, 42, 51, 60, 34, 35, 36, 37, 38, 39, 24, 25, 26, 17, 9, 19, 12]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(33, field), f'From 33 to {field}'
        else:
            assert not game.check_move_rules(33, field), f'From 33 to {field}'

    good_fields = [59, 60, 61, 53, 51, 50,
                   45, 38, 31,
                   16, 25, 34, 43,
                   44, 36, 28]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(52, field), f'From 52 to {field}'
        else:
            assert not game.check_move_rules(52, field), f'From 52 to {field}'

    good_fields = [61, 62, 53, 55, 47,
                   46, 38, 30, 22, 14, 6,
                   45, 36, 27, 18, 9]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(54, field), f'From 54 to {field}'
        else:
            assert not game.check_move_rules(54, field), f'From 54 to {field}'


def test_king_move():
    game = ChessGame(empty_board=True)
    game.add_figure(13, 'king', 'white')
    game.add_figure(29, 'king', 'black')

    game.add_figure(2, 'king', 'white')
    game.add_figure(10, 'pawn', 'white')

    print_board_state(game)

    good_fields = [12, 4, 5, 6, 14]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(13, field), f"From 13 to {field}"
        else:
            assert not game.check_move_rules(13, field), f"From 13 to {field}"

    good_fields = [28, 30, 38, 37, 36]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(29, field), f"From 29 to {field}"
        else:
            assert not game.check_move_rules(29, field), f"From 29 to {field}"

    good_fields = [1, 9, 11, 3]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(2, field), f"From 2 to {field}"
        else:
            assert not game.check_move_rules(2, field), f"From 2 to {field}"


def test_king_no_moves():
    game = ChessGame()
    print_board_state(game)

    for field in range(64):
        assert not game.check_move_rules(4, field), f"From 4 to {field}"


def test_castling():
    game = ChessGame()
    for x in range(8, 16):
        game.clear_pos(x)

    game.clear_pos(1)
    game.clear_pos(2)
    game.clear_pos(3)
    game.clear_pos(5)
    game.clear_pos(6)

    print_board_state(game)

    good_fields = [2, 3, 11, 12, 13, 5, 6]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(4, field), f"From 4 to {field}"
        else:
            assert not game.check_move_rules(4, field), f"From 4 to {field}"


def test_castling_check():
    game = ChessGame()
    game.add_figure(44, 'rook', 'black')

    for x in range(8, 16):
        game.clear_pos(x)

    game.clear_pos(1)
    game.clear_pos(2)
    game.clear_pos(3)
    game.clear_pos(5)
    game.clear_pos(6)

    print_board_state(game)

    good_fields = [3, 11, 13, 5]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(4, field), f"From 4 to {field}"
        else:
            assert not game.check_move_rules(4, field), f"From 4 to {field}"


def test_castling_attack_line():
    game = ChessGame()
    game.add_figure(43, 'rook', 'black')
    game.add_figure(45, 'rook', 'black')

    for x in range(8):
        game.board[1, x] = 8 + x
    game.board[0, 1] = 1
    game.board[0, 2] = 2
    game.board[0, 3] = 3
    game.board[0, 5] = 5
    game.board[0, 6] = 6

    print_board_state(game)

    good_fields = [12]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(4, field)
        else:
            assert not game.check_move_rules(4, field)


def test_castling_attack_line_black():
    game = ChessGame()
    game.add_figure(43, 'rook', 'white')
    game.add_figure(46, 'rook', 'white')

    for x in range(48, 56):
        game.clear_pos(x)
    game.clear_pos(61)
    game.clear_pos(62)
    print_board_state(game)

    good_fields = [52, 61, 53]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(60, field), f"From 60 to {field}"
        else:
            assert not game.check_move_rules(60, field), f"From 60 to {field}"


def test_castling_queen_block():
    game = ChessGame()

    for x in range(8, 17):
        game.clear_pos(x)
    game.clear_pos(1)
    game.clear_pos(2)
    game.clear_pos(5)
    game.clear_pos(6)

    print_board_state(game)

    good_fields = [11, 12, 13, 5, 6]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(4, field), f"From 4 to {field}"
        else:
            assert not game.check_move_rules(4, field), f"From 4 to {field}"


def test_castling_bishops_block():
    game = ChessGame()

    for x in range(8):
        game.board[1, x] = 8 + x
    game.clear_pos(1)
    game.clear_pos(3)
    game.clear_pos(6)

    print_board_state(game)

    good_fields = [3, 11, 12, 13]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(4, field), f"From 4 to {field}"
        else:
            assert not game.check_move_rules(4, field), f"From 4 to {field}"


def test_castling_bishops_block_black():
    game = ChessGame()

    for x in range(48, 56):
        game.clear_pos(x)
    game.clear_pos(59)
    game.clear_pos(58)
    game.clear_pos(62)

    print_board_state(game)

    good_fields = [59, 51, 52, 53]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(60, field), f"From 60 to {field}"
        else:
            assert not game.check_move_rules(60, field), f"From 60 to {field}"


def test_castling_from_different():
    game = ChessGame(empty_board=True)
    game.add_figure(0, 'rook', 'white')
    game.add_figure(7, 'rook', 'white')
    game.add_figure(12, 'king', 'white')
    print_board_state(game)

    good_fields = [3, 11, 19, 20, 21, 13, 5, 4]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(12, field), f"From 12 to {field}"
        else:
            assert not game.check_move_rules(12, field), f"From 12 to {field}"


def test_castling_after_move():
    game = ChessGame(empty_board=True)
    game.add_figure(0, 'rook', 'white')
    game.add_figure(7, 'rook', 'white')
    game.add_figure(4, 'king', 'white')

    game.add_figure(56, 'rook', 'white')
    game.add_figure(63, 'rook', 'white')
    game.add_figure(60, 'king', 'white')

    game.make_move(4, 12)
    game.make_move(12, 4)

    game.make_move(60, 51)
    game.make_move(51, 60)

    print("Moved all kings")
    print_board_state(game)

    good_fields = [3, 11, 12, 13, 5]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(4, field), f"From 4 to {field}"
        else:
            assert not game.check_move_rules(4, field), f"From 4 to {field}"

    good_fields = [59, 51, 52, 53, 61]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(60, field), f"From 60 to {field}"
        else:
            assert not game.check_move_rules(60, field), f"From 60 to {field}"


def test_castling_after_rook_move():
    game = ChessGame(empty_board=True)
    game.add_figure(0, 'rook', 'white')
    game.add_figure(7, 'rook', 'white')
    game.add_figure(4, 'king', 'white')

    game.add_figure(56, 'rook', 'white')
    game.add_figure(63, 'rook', 'white')
    game.add_figure(60, 'king', 'white')

    game.make_move(0, 1)
    game.make_move(1, 0)
    game.make_move(7, 15)
    game.make_move(15, 7)

    game.make_move(56, 48)
    game.make_move(48, 56)
    game.make_move(63, 55)
    game.make_move(55, 63)

    print("Moved all rooks")
    print_board_state(game)

    good_fields = [3, 11, 12, 13, 5]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(4, field), f"From 4 to {field}"
        else:
            assert not game.check_move_rules(4, field), f"From 4 to {field}"

    good_fields = [59, 51, 52, 53, 61]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(60, field), f"From 60 to {field}"
        else:
            assert not game.check_move_rules(60, field), f"From 60 to {field}"


def test_castling_no_rook():
    game = ChessGame(empty_board=True)
    game.add_figure(4, 'king', 'white')
    game.add_figure(60, 'king', 'black')

    print_board_state(game)

    good_fields = [3, 11, 12, 13, 5]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(4, field), f"From 4 to {field}"
        else:
            assert not game.check_move_rules(4, field), f"From 4 to {field}"

    good_fields = [59, 51, 52, 53, 61]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(60, field), f"From 60 to {field}"
        else:
            assert not game.check_move_rules(60, field), f"From 60 to {field}"


def test_success_castling_1():
    game = ChessGame(empty_board=True)
    game.add_figure(0, 'rook', 'white')
    game.add_figure(7, 'rook', 'white')
    game.add_figure(4, 'king', 'white')
    print_board_state(game)

    good_fields = [2, 6, 3, 11, 12, 13, 5]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(4, field), f"From 4 to {field}"
        else:
            assert not game.check_move_rules(4, field), f"From 4 to {field}"

    game.make_move(4, 6)
    assert game.board[0, 5].fig_type == 'rook'
    assert type(game.board[0, 7]) is int


def test_success_castling_2():
    game = ChessGame(empty_board=True)
    game.add_figure(0, 'rook', 'white')
    game.add_figure(7, 'rook', 'white')
    game.add_figure(4, 'king', 'white')
    print_board_state(game)

    good_fields = [2, 6, 3, 11, 12, 13, 5]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(4, field), f"From 4 to {field}"
        else:
            assert not game.check_move_rules(4, field), f"From 4 to {field}"

    game.make_move(4, 2)
    assert game.board[0, 3].fig_type == 'rook'
    assert type(game.board[0, 0]) is int


def test_success_castling_3_black():
    game = ChessGame(empty_board=True)
    game.add_figure(60, 'king', 'black')
    game.add_figure(56, 'rook', 'black')
    game.add_figure(63, 'rook', 'black')
    print_board_state(game)

    good_fields = [58, 59, 51, 52, 53, 61, 62]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(60, field), f"From 60 to {field}"
        else:
            assert not game.check_move_rules(60, field), f"From 60 to {field}"

    game.make_move(60, 58)
    assert game.board[7, 3].fig_type == 'rook'
    assert type(game.board[7, 0]) is int


def test_success_castling_4_black():
    game = ChessGame(empty_board=True)
    game.add_figure(60, 'king', 'black')
    game.add_figure(56, 'rook', 'black')
    game.add_figure(63, 'rook', 'black')
    print_board_state(game)

    good_fields = [58, 59, 51, 52, 53, 61, 62]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(60, field), f"From 60 to {field}"
        else:
            assert not game.check_move_rules(60, field), f"From 60 to {field}"

    game.make_move(60, 62)
    assert game.board[7, 5].fig_type == 'rook'
    assert type(game.board[7, 7]) is int


def test_king_check():
    game = ChessGame()
    for x in range(8, 17):
        game.clear_pos(x)
    game.clear_pos(1)
    game.clear_pos(2)
    game.clear_pos(3)
    game.clear_pos(5)
    game.clear_pos(6)
    game.add_figure(8, 'rook', 'black')
    game.add_figure(15, 'rook', 'black')
    game.add_figure(43, 'rook', 'black')
    game.add_figure(44, 'rook', 'black')
    game.add_figure(45, 'rook', 'black')

    game.add_figure(20, 'queen', 'white')
    print_board_state(game)

    for field in range(64):
        assert not game.check_move_rules(4, field), f"From 4 to {field}"

    good_fields = [12, 28, 36, 44]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(20, field), f"From 20 to {field}"
        else:
            assert not game.check_move_rules(20, field), f"From 20 to {field}"


def test_king_check_2():
    game = ChessGame()
    game.add_figure(31, 'bishop', 'black')
    game.add_figure(20, 'pawn', 'black')
    print_board_state(game)

    for field in range(64):
        assert not game.check_move_rules(13, field)


def test_king_attack():
    game = ChessGame(empty_board=True)
    game.add_figure(25, 'bishop', 'black')
    game.add_figure(4, 'king', 'white')
    game.add_figure(11, 'pawn', 'black')
    game.add_figure(13, 'pawn', 'black')
    print_board_state(game)

    good_fields = [3, 12, 13, 5]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(4, field)
        else:
            assert not game.check_move_rules(4, field)


def test_king_attack_blocked_by_king():
    game = ChessGame(empty_board=True)
    game.add_figure(4, 'king', 'white')

    game.add_figure(11, 'pawn', 'black')
    game.add_figure(13, 'pawn', 'black')
    game.add_figure(25, 'bishop', 'black')
    game.add_figure(21, 'king', 'black')
    print_board_state(game)

    good_fields = [3, 5]
    for field in range(64):
        if field in good_fields:
            assert game.check_move_rules(4, field)
        else:
            assert not game.check_move_rules(4, field)


def test_moves_left_for_white():
    game = ChessGame(empty_board=True)
    game.add_figure(4, 'king', 'white')
    game.add_figure(10, 'rook', 'black')
    game.add_figure(27, 'rook', 'black')
    game.add_figure(29, 'rook', 'black')
    print_board_state(game)

    assert not game.check_moves_left()

    game.add_figure(15, 'pawn', 'white')
    print_board_state(game)
    assert game.check_moves_left()

    game.clear_pos(15)
    game.add_figure(28, 'pawn', 'white')
    print_board_state(game)
    assert game.check_moves_left()

    game.clear_pos(28)
    game.add_figure(22, 'rook', 'white')
    game.add_figure(31, 'bishop', 'black')
    print_board_state(game)
    assert not game.check_moves_left()


def test_cant_kill_king():
    game = ChessGame()
    game.add_figure(27, 'king', 'white')

    game.add_figure(9, 'bishop', 'black')
    game.add_figure(25, 'rook', 'black')
    game.add_figure(29, 'queen', 'black')
    game.add_figure(34, 'knight', 'black')
    game.add_figure(36, 'pawn', 'black')
    print_board_state(game)

    figures = [9, 25, 29, 34, 36]
    for figure in figures:
        assert not game.check_move_rules(figure, 27), f"From {figure} to 27`"


def test_move_invalid_figure():
    game = ChessGame()
    print_board_state(game)
    assert not game.check_move_rules(24, 25), "From 24 to 25"
    assert not game.check_move_rules(16, 75), "From 16 to 75"
    assert not game.check_move_rules(-10, 15), "From -10 to 15"
    assert not game.check_move_rules(0, 0), "From 0 to 0"


def test_state_after_move():
    game = ChessGame(empty_board=True)
    game.add_figure(4, 'king', 'white')
    game.add_figure(10, 'queen', 'white')
    game.add_figure(12, 'queen', 'white')
    game.add_figure(44, 'rook', 'black')
    print_board_state(game)

    game.make_move(12, 21)
    assert game.board[1, 4].fig_type == 'queen'

    print_board_state(game)
    game.check_move_rules(10, 24)
    assert game.board[1, 2].fig_type == 'queen'

    game.make_move(10, 24)
    assert type(game.board[1, 2]) is int
    assert game.board[3, 0].fig_type == 'queen'


def print_board_state(game):
    print(f"Board before assert:\nBlack\n{game.pretty_state()}\nWhite")


def print_board_state_flipped(game):
    print(f"Initial board FLIPPED:\nWhite\n{game.getstate()}\nBlack")
