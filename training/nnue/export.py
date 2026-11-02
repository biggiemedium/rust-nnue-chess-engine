# Quantize the trained net and write the file engine/src/nnue/network.rs loads
#
# floats -> ints so the engine can do integer maths:
#   feature transformer * QA
#   output weights      * QB
#   output bias         * QA * QB  (it gets added after the two scales multiply)
#
#   python -m nnue.export runs/net-001/checkpoints/best.pt runs/net-001/net-001.nnue

import argparse
import struct

import numpy as np
import torch

from .features import INPUTS
from .model import NNUE

MAGIC = b"NNUE"
VERSION = 1
QA = 255
QB = 64


def quantize(model):
    ft_w = model.ft.weight.detach()[:INPUTS]  # drop the PAD row
    ft_b = model.ft_bias.detach()
    out_w = model.out.weight.detach()[0]
    out_b = model.out.bias.detach()[0]

    def to_i16(t, scale):
        q = torch.round(t * scale)
        if q.abs().max() > 32767:
            raise ValueError("weight overflowed i16 after quantizing, clip_weights() not run?")
        return q.to(torch.int16).numpy()

    return {
        "ft_weights": to_i16(ft_w, QA),
        "ft_bias": to_i16(ft_b, QA),
        "out_weights": to_i16(out_w, QB),
        "out_bias": int(torch.round(out_b * QA * QB)),
    }


def write(model, path):
    q = quantize(model)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, model.hidden))
        f.write(q["ft_weights"].astype("<i2").tobytes())
        f.write(q["ft_bias"].astype("<i2").tobytes())
        f.write(q["out_weights"].astype("<i2").tobytes())
        f.write(struct.pack("<i", q["out_bias"]))


def quantized_eval(q, stm, nstm):
    # integer forward pass, same maths the engine will do, used to test export is right
    # stm / nstm = lists of active feature indices
    def accumulate(features):
        acc = q["ft_bias"].astype(np.int32).copy()
        for f in features:
            acc += q["ft_weights"][f]
        return acc

    hidden = len(q["ft_bias"])
    acc = np.concatenate([accumulate(stm), accumulate(nstm)])
    act = np.clip(acc, 0, QA) ** 2  # SCReLU in ints, scale is now QA * QA
    out = int(np.dot(act.astype(np.int64), q["out_weights"].astype(np.int64)))
    out = out // QA + q["out_bias"]  # back to QA * QB
    assert len(acc) == 2 * hidden
    return out * 400 // (QA * QB)  # centipawns


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    model = NNUE(hidden=ckpt["config"]["model"]["hidden"])
    model.load_state_dict(ckpt["model"])
    return model


def main():
    parser = argparse.ArgumentParser(description="export a checkpoint to the engine's .nnue format")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()

    write(load_checkpoint(args.checkpoint), args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
