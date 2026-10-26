import torch
from torch import nn
import torch.nn.functional as F


class Combiner(nn.Module):
    def __init__(self, combine_config={}):
        super(Combiner, self).__init__()
        self.input_dim = combine_config.get('input_dim', 0)
        self.dim = combine_config.get('dim', 512)
        self.models = combine_config.get('models', 'lstm')

        if self.models == 'lstm':
            self.lstm_dropout = combine_config.get('lstm_dropout', 0.1)
            self.rnn_dim = combine_config.get('rnn_dim', 512)
            self.num_layers = combine_config.get('num_layers', 2)
            self.lstm = nn.LSTM(self.input_dim, self.rnn_dim, self.num_layers,
                                batch_first=True, bidirectional=True,
                                dropout=self.lstm_dropout if self.num_layers > 1 else 0)
            self.output_proj = nn.Linear(self.rnn_dim * 2, self.dim)
        elif self.models == 'naive':
            self.output_proj = nn.Linear(self.input_dim, self.dim)
        elif self.models == 'reformer':
            from performer_pytorch import SelfAttention
            self.self_attention = SelfAttention(dim=self.input_dim, heads=8, dim_head=64)
            self.output_proj = nn.Linear(self.input_dim, self.dim)

    def forward(self, hidden, word_mask):
        if self.models == 'lstm':
            output, (h_n, c_n) = self.lstm(hidden)
            output = output[:, :, :self.rnn_dim] + output[:, :, self.rnn_dim:]
            output = self.output_proj(output)
            return output
        elif self.models == 'naive':
            return self.output_proj(hidden)
        elif self.models == 'reformer':
            output = self.self_attention(hidden)
            return self.output_proj(output)


def create_combiner(config):
    models = config.get('models', 'lstm')
    if models == 'lstm':
        return Combiner(config)
    elif models == 'naive':
        return NaiveCombiner(config)
    elif models == 'reformer':
        return ReformerCombiner(config)
    else:
        return Combiner(config)


class NaiveCombiner(nn.Module):
    def __init__(self, config={}):
        super(NaiveCombiner, self).__init__()
        self.input_dim = config.get('input_dim', 0)
        self.dim = config.get('dim', 512)
        self.proj = nn.Linear(self.input_dim, self.dim)

    def forward(self, hidden, word_mask):
        return self.proj(hidden)


class ReformerCombiner(nn.Module):
    def __init__(self, config={}):
        super(ReformerCombiner, self).__init__()
        from performer_pytorch import SelfAttention
        self.input_dim = config.get('input_dim', 0)
        self.dim = config.get('dim', 512)
        self.self_attention = SelfAttention(dim=self.input_dim, heads=8, dim_head=64)
        self.proj = nn.Linear(self.input_dim, self.dim)

    def forward(self, hidden, word_mask):
        output = self.self_attention(hidden)
        return self.proj(output)