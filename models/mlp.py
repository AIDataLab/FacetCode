import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0, act='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.act_fn = self.get_act_fn(act)

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            if i == num_layers - 1:
                out_dim = output_dim
            else:
                out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(self.act_fn)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.layers = nn.Sequential(*layers)

    def get_act_fn(self, act):
        if act == 'relu':
            return nn.ReLU()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'gelu':
            return nn.GELU()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU()

    def forward(self, x):
        return self.layers(x)