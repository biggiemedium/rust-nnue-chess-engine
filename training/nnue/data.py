# Reads the 32 byte records the engine writes (engine/src/datagen/format.rs has the layout)
# the file is memory mapped so a few GB of positions never has to fit in RAM

from pathlib import Path

import numpy as np
import torch

from .features import MAX_PIECES, PAD, feature_index

RECORD = np.dtype([
    ("occupancy", "<u8"),
    ("pieces", "u1", 16),
    ("score", "<i2"),  # centipawns, side to move POV
    ("result", "u1"),  # 0 loss, 1 draw, 2 win, side to move POV
    ("stm", "u1"),  # 0 white, 1 black
    ("reserved", "u1", 4),
])
assert RECORD.itemsize == 32

_SQUARES = np.arange(64, dtype=np.uint64)


def load(path):
    path = Path(path)
    if path.stat().st_size % RECORD.itemsize != 0:
        raise ValueError(f"{path} is not a whole number of 32 byte records")
    return np.memmap(path, dtype=RECORD, mode="r")


def decode(records):
    # records -> (stm features, opponent features), both (N, 32) padded with PAD
    n = len(records)

    # unpack occupancy into (N, 64) bools, np.nonzero walks it row by row, lowest square first
    # which is the same order the engine packed the pieces in
    bits = ((records["occupancy"][:, None] >> _SQUARES) & 1).astype(bool)
    rows, squares = np.nonzero(bits)

    # ordinal = which piece this is within its own row (0, 1, 2 ...)
    counts = bits.sum(axis=1)
    starts = np.cumsum(counts) - counts
    ordinal = np.arange(len(rows)) - np.repeat(starts, counts)

    # two nibbles per byte, even ordinal in the low half
    byte = records["pieces"][rows, ordinal // 2]
    nibble = np.where(ordinal % 2 == 0, byte & 0xF, byte >> 4).astype(np.int64)
    colour = nibble >> 3
    piece_type = nibble & 7

    stm = records["stm"][rows].astype(np.int64)
    squares = squares.astype(np.int64)

    stm_features = np.full((n, MAX_PIECES), PAD, dtype=np.int64)
    nstm_features = np.full((n, MAX_PIECES), PAD, dtype=np.int64)
    stm_features[rows, ordinal] = feature_index(stm, colour, piece_type, squares)
    nstm_features[rows, ordinal] = feature_index(1 - stm, colour, piece_type, squares)
    return stm_features, nstm_features


def batches(records, batch_size, shuffle=True, rng=None):
    # yields torch tensors ready for the model, one shuffled pass over the data
    order = np.arange(len(records))
    if shuffle:
        (rng or np.random.default_rng()).shuffle(order)

    for start in range(0, len(order), batch_size):
        # sorting the indices makes the memmap reads way less random
        idx = np.sort(order[start:start + batch_size])
        chunk = records[idx]
        stm, nstm = decode(chunk)
        yield (
            torch.from_numpy(stm),
            torch.from_numpy(nstm),
            torch.from_numpy(chunk["score"].astype(np.float32)),
            torch.from_numpy(chunk["result"].astype(np.float32) / 2),  # 0, 0.5, 1
        )


def split(records, val_fraction):
    # last slice of the file is validation, the engine writes games in order
    # so this keeps whole games on one side (no position leaking into both)
    cut = int(len(records) * (1 - val_fraction))
    return records[:cut], records[cut:]


# --- writing, the engine does this for real data, python only needs it for tests + tools ---

def encode(board, score, result):
    # board is a python-chess Board, score/result from side to move's POV
    out = np.zeros(1, dtype=RECORD)
    occupancy = int(board.occupied)
    out["occupancy"] = occupancy

    pieces = np.zeros(16, dtype=np.uint8)
    i = 0
    for sq in range(64):
        if not occupancy >> sq & 1:
            continue
        piece = board.piece_at(sq)
        colour = 0 if piece.color else 1  # python-chess: True = white
        nibble = colour << 3 | (piece.piece_type - 1)  # python-chess pawn = 1
        pieces[i // 2] |= nibble if i % 2 == 0 else nibble << 4
        i += 1

    out["pieces"] = pieces
    out["score"] = score
    out["result"] = result
    out["stm"] = 0 if board.turn else 1
    return out.tobytes()


def write(path, records):
    # records = iterable of encode() bytes
    with open(path, "wb") as f:
        for r in records:
            f.write(r)
