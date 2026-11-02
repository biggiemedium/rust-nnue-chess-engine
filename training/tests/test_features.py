# same numbers as the tests in engine/src/nnue/features.rs, if one side changes the other has to

from nnue.features import feature_index

WHITE, BLACK = 0, 1
PAWN, KING = 0, 5


def test_white_pawn_e2():
    assert feature_index(WHITE, WHITE, PAWN, 12) == 12
    assert feature_index(BLACK, WHITE, PAWN, 12) == 384 + 52


def test_black_king_e8():
    assert feature_index(BLACK, BLACK, KING, 60) == 5 * 64 + 4
    assert feature_index(WHITE, BLACK, KING, 60) == 384 + 5 * 64 + 60
