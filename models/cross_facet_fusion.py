import torch
from torch import nn
import torch.nn.functional as F
from constant import K_FACETS


class CrossFacetFusion(nn.Module):
    def __init__(self, hidden_dim=512, num_facets=K_FACETS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_facets = num_facets

        self.channel_attention = nn.Sequential(
            nn.Linear(hidden_dim * num_facets, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_facets),
            nn.Softmax(dim=-1)
        )

        self.facet_transform = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_facets)
        ])

        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, document_hidden, code_facets):
        batch_size = document_hidden.size(0)

        evidence_vectors = []
        for k in range(self.num_facets):
            facet = code_facets[:, k, :]
            transformed_facet = self.facet_transform[k](facet)

            affinity = torch.sum(document_hidden * transformed_facet.unsqueeze(1), dim=-1)
            attention_weights = F.softmax(affinity, dim=1)

            evidence = torch.sum(document_hidden * attention_weights.unsqueeze(-1), dim=1)
            evidence_vectors.append(evidence)

        evidence_stack = torch.stack(evidence_vectors, dim=1)

        channel_weights = self.channel_attention(evidence_stack.reshape(batch_size, -1))
        channel_weights = channel_weights.unsqueeze(-1)

        weighted_evidence = (evidence_stack * channel_weights).sum(dim=1)

        max_pooled = evidence_stack.max(dim=1)[0]

        final_representation = torch.cat([weighted_evidence, max_pooled], dim=-1)
        output = self.output_projection(final_representation)

        return output