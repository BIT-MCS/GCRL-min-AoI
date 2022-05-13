import torch
import torch.nn as nn


def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        mlp_i=nn.Linear(mlp_dims[i], mlp_dims[i + 1])
        nn.init.xavier_uniform_(mlp_i.weight)
        nn.init.constant_(mlp_i.bias, 0.0)
        layers.append(mlp_i)
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    # print(net)
    return net
