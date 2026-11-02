import struct

import torch

from nnue import export
from nnue.features import INPUTS, PAD
from nnue.model import NNUE


def test_quantized_eval_matches_float_model():
    # the int maths the engine will do should land within a few cp of the float net
    torch.manual_seed(0)
    model = NNUE(hidden=64)
    with torch.no_grad():
        model.out.weight.uniform_(-1, 1)  # bigger than init so the output isn't ~0
    q = export.quantize(model)

    for _ in range(50):
        stm = torch.randint(0, INPUTS, (1, 32))
        nstm = torch.randint(0, INPUTS, (1, 32))
        float_cp = model(stm, nstm).item() * 400
        int_cp = export.quantized_eval(q, stm[0].tolist(), nstm[0].tolist())
        assert abs(float_cp - int_cp) < 10, (float_cp, int_cp)


def test_file_layout(tmp_path):
    model = NNUE(hidden=32)
    path = tmp_path / "net.nnue"
    export.write(model, path)

    raw = path.read_bytes()
    assert raw[:4] == b"NNUE"
    assert struct.unpack("<II", raw[4:12]) == (export.VERSION, 32)
    # header + ft weights + ft bias + out weights (i16) + out bias (i32)
    assert len(raw) == 12 + (INPUTS * 32 + 32 + 64) * 2 + 4


def test_pad_row_stays_zero():
    model = NNUE(hidden=16)
    assert torch.all(model.ft.weight[PAD] == 0)
