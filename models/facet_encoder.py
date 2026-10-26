import torch
from torch import nn
import torch.nn.functional as F
from models.mlp import MLP
from constant import FACET_DIMENSIONS, K_FACETS


class FacetEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, num_facets=K_FACETS):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_facets = num_facets

        self.facet_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, input_dim)
            for _ in range(num_facets)
        ])

        self.facet_bilstm = nn.ModuleList([
            nn.LSTM(input_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
            for _ in range(num_facets)
        ])

        self.facet_projection = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_facets)
        ])

    def forward(self, facet_texts):
        outputs = []
        for i in range(self.num_facets):
            embedded = self.facet_embeddings[i](facet_texts[i])
            output, _ = self.facet_bilstm[i](embedded)
            pooled = output.max(dim=1)[0]
            projected = self.facet_projection[i](pooled)
            outputs.append(projected)
        return torch.stack(outputs, dim=1)


class MultiDimensionalRepresentation(nn.Module):
    def __init__(self, input_dim=512, num_facets=K_FACETS):
        super().__init__()
        self.num_facets = num_facets
        self.facet_nn = nn.ModuleList([
            nn.Linear(input_dim, input_dim)
            for _ in range(num_facets)
        ])

    def forward(self, base_representations):
        expanded = []
        for i in range(self.num_facets):
            expanded.append(self.facet_nn[i](base_representations))
        return torch.stack(expanded, dim=1)


class CachedFacetStore:
    def __init__(self):
        self.facet_representations = {}
        self.ancestor_representations = {}

    def store(self, code, representations, ancestors=None):
        self.facet_representations[code] = representations
        if ancestors:
            for ancestor in ancestors:
                if ancestor not in self.ancestor_representations:
                    self.ancestor_representations[ancestor] = []
                self.ancestor_representations[ancestor].append(representations)

    def get(self, code):
        return self.facet_representations.get(code, None)

    def get_ancestor(self, ancestor_code):
        return self.ancestor_representations.get(ancestor_code, None)