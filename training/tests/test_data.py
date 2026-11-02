import chess
import numpy as np

from nnue import data
from nnue.features import PAD, feature_index

from conftest import random_positions

# same golden bytes as engine/src/datagen/format.rs, proves rust and python agree on the format
STARTPOS_DRAW = bytes([
    0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF,
    0x13, 0x42, 0x25, 0x31,
    0x00, 0x00, 0x00, 0x00,
    0x88, 0x88, 0x88, 0x88,
    0x9B, 0xCA, 0xAD, 0xB9,
    0x00, 0x00,
    0x01,
    0x00,
    0x00, 0x00, 0x00, 0x00,
])


def test_startpos_matches_engine_golden():
    assert data.encode(chess.Board(), 0, 1) == STARTPOS_DRAW


def expected_features(board, perspective):
    # slow but obviously right version straight from python-chess
    out = []
    for sq, piece in board.piece_map().items():
        colour = 0 if piece.color else 1
        out.append(int(feature_index(perspective, colour, piece.piece_type - 1, sq)))
    return sorted(out)


def test_decode_matches_python_chess(tmp_path):
    boards = random_positions(200)
    path = tmp_path / "p.bin"
    data.write(path, (data.encode(b, 0, 1) for b in boards))

    stm, nstm = data.decode(data.load(path))
    for i, board in enumerate(boards):
        us = 0 if board.turn else 1
        assert sorted(f for f in stm[i] if f != PAD) == expected_features(board, us)
        assert sorted(f for f in nstm[i] if f != PAD) == expected_features(board, 1 - us)


def test_score_result_stm_roundtrip(tmp_path):
    board = chess.Board()
    board.push_san("e4")  # black to move
    path = tmp_path / "p.bin"
    data.write(path, [data.encode(board, -123, 2)])

    r = data.load(path)[0]
    assert (r["score"], r["result"], r["stm"]) == (-123, 2, 1)


def test_batches_cover_every_position_once(dataset):
    records = data.load(dataset)
    seen = sum(len(score) for _, _, score, _ in data.batches(records, 512, rng=np.random.default_rng(0)))
    assert seen == len(records)
