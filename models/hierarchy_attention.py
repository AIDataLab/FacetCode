import torch
from torch import nn
import torch.nn.functional as F
import math


class HierarchyGuidedAttention(nn.Module):
    def __init__(self, hidden_dim=512, attention_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        self.summary_proj = nn.Linear(hidden_dim, attention_dim)
        self.context_proj = nn.Linear(hidden_dim, attention_dim)

        self.att_score = nn.Linear(attention_dim * 2, attention_dim)
        self.att_out = nn.Linear(attention_dim, 1)

        self.softmax = nn.Softmax(dim=-1)

    def compute_taxonomic_affinity(self, code_summary, ancestor_summary):
        code_proj = self.summary_proj(code_summary)
        ancestor_proj = self.summary_proj(ancestor_summary)

        combined = torch.cat([code_proj, ancestor_proj], dim=-1)
        score = torch.tanh(self.att_score(combined))
        score = self.att_out(score)
        return score.squeeze(-1)

    def forward(self, code_representations, ancestor_representations_list):
        code_summary = code_representations[:, -1, :]

        ancestor_weights = []
        for ancestor_repr in ancestor_representations_list:
            weight = self.compute_taxonomic_affinity(code_summary, ancestor_repr[:, -1, :])
            ancestor_weights.append(weight)

        ancestor_weights = torch.stack(ancestor_weights, dim=-1)
        ancestor_weights = self.softmax(ancestor_weights)

        aggregated_ancestors = []
        for i, ancestor_repr in enumerate(ancestor_representations_list):
            weight = ancestor_weights[:, i].unsqueeze(-1)
            aggregated = (ancestor_repr * weight.unsqueeze(-1)).sum(dim=1)
            aggregated_ancestors.append(aggregated)

        final_representations = []
        for i, code_repr in enumerate(code_representations.split(1, dim=1)):
            code_repr = code_repr.squeeze(1)
            combined = code_repr + aggregated_ancestors[i]
            final_representations.append(combined)

        return torch.stack(final_representations, dim=1)