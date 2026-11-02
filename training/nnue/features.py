# NNUE input features, mirror of engine/src/nnue/features.rs
# if these two disagree the engine reads the net wrong, test_features.py checks the same numbers as the rust tests
#
# 768 inputs = 2 sides * 6 piece types * 64 squares
# own pieces -> 0..384, enemy pieces -> 384..768, black flips the board (sq ^ 56)

import numpy as np

INPUTS = 768
PAD = INPUTS  # extra index for empty slots, the embedding row for it is always 0
MAX_PIECES = 32


def feature_index(perspective, colour, piece_type, square):
    # works on plain ints or whole numpy arrays at once
    side = np.where(colour == perspective, 0, 1)
    sq = np.where(perspective == 0, square, square ^ 56)
    return side * 384 + piece_type * 64 + sq
