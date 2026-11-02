# (768 -> hidden) x 2 perspectives -> SCReLU -> 1
# https://www.chessprogramming.org/NNUE

import math

import torch
from torch import nn

from .features import INPUTS, PAD

# the engine stores weights as ints, they have to fit after scaling (see export.py)
# out weights * QB(64) must stay inside i8 range to leave headroom -> 127 / 64
OUT_WEIGHT_CLIP = 127 / 64


def screlu(x):
    # squared clipped relu, clamp to [0, 1] then square
    return torch.clamp(x, 0, 1) ** 2


class NNUE(nn.Module):

    def __init__(self, hidden=256):
        super().__init__()
        self.hidden = hidden

        # feature transformer, an embedding bag is just a sparse linear layer:
        # sum of the weight rows for every active feature, PAD row is skipped
        self.ft = nn.EmbeddingBag(INPUTS + 1, hidden, mode="sum", padding_idx=PAD)
        self.ft_bias = nn.Parameter(torch.zeros(hidden))
        self.out = nn.Linear(2 * hidden, 1)

        # same init a Linear(768, hidden) would get, default embedding init (std 1) is way too big
        bound = 1 / math.sqrt(INPUTS)
        with torch.no_grad():
            self.ft.weight.uniform_(-bound, bound)
            self.ft.weight[PAD].zero_()

    def forward(self, stm, nstm):
        # same weights for both sides, only the order of the halves changes
        a = self.ft(stm) + self.ft_bias
        b = self.ft(nstm) + self.ft_bias
        x = screlu(torch.cat([a, b], dim=1))
        return self.out(x).squeeze(1)  # 1.0 == EVAL_SCALE centipawns

    def clip_weights(self):
        # run after every optimizer step so quantization never overflows
        with torch.no_grad():
            self.out.weight.clamp_(-OUT_WEIGHT_CLIP, OUT_WEIGHT_CLIP)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
