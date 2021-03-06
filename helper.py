import torch
from torch import nn


def init_weights(model, size=100):
    weight = torch.FloatTensor(size, size).normal_(0, 1)
    weight = 5.7 * weight / torch.norm(weight)
    model.linear.weight = nn.Parameter(weight)
    return model


def create_input(dim, size=100):
    return torch.FloatTensor(dim, size).normal_(0, 1)
