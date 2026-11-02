import random

import chess
import pytest

from nnue import data


def random_positions(n, seed=0):
    # random games, good enough to test the plumbing (not for training a real net)
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = chess.Board()
        for _ in range(rng.randint(0, 80)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        out.append(board.copy())
    return out


@pytest.fixture
def dataset(tmp_path):
    # small .bin file in the engine's format
    rng = random.Random(1)
    path = tmp_path / "positions.bin"
    data.write(path, (data.encode(b, rng.randint(-600, 600), rng.randint(0, 2)) for b in random_positions(3000)))
    return path
