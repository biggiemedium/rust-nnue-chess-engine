# Loss = MSE in "win probability" space
#
# score -> win prob with a sigmoid: sigmoid(cp / 400) (same curve Elo uses)
# target blends what search thought (score) with what actually happened (result):
#   target = lambda * result + (1 - lambda) * sigmoid(score / 400)
# lambda = 0 -> pure search score, lambda = 1 -> pure game result

import torch

EVAL_SCALE = 400


def wdl_loss(output, score, result, wdl_lambda):
    prediction = torch.sigmoid(output)  # output is already in EVAL_SCALE units
    target = wdl_lambda * result + (1 - wdl_lambda) * torch.sigmoid(score / EVAL_SCALE)
    return torch.mean((prediction - target) ** 2)
