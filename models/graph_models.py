import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract


class KGCodeReassign(nn.Module):
    def __init__(self, config, edges_dict, c2ind, mc2ind):
        super().__init__()
        self.hidden_dim = config['attention_dim']
        self.edge_dim = config.get('edge_dim', 64)

        self.c2ind = c2ind
        self.mc2ind = mc2ind

        self.code_count = len(c2ind)
        self.main_code_count = len(mc2ind)

        edges = self.build_edges(edges_dict)
        self.edge_index = edges['edge_index']
        self.edge_type = edges['edge_type']
        self.edge_weight = edges['edge_weight']

        self.edge_embedding = nn.Embedding(self.edge_index.size(0), self.edge_dim)

        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)

    def build_edges(self, edges_dict):
        edge_list = []
        edge_types = []
        edge_weights = []

        for code, neighbors in edges_dict.items():
            if code not in self.c2ind:
                continue
            code_idx = self.c2ind[code]

            for neighbor_info in neighbors:
                if isinstance(neighbor_info, tuple):
                    neighbor_code, relation_type, weight = neighbor_info
                else:
                    neighbor_code = neighbor_info
                    relation_type = 0
                    weight = 1.0

                if neighbor_code in self.c2ind:
                    neighbor_idx = self.c2ind[neighbor_code]
                    edge_list.append((code_idx, neighbor_idx))
                    edge_types.append(relation_type)
                    edge_weights.append(weight)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)

        return {
            'edge_index': edge_index,
            'edge_type': edge_type,
            'edge_weight': edge_weight
        }

    def forward(self, node_features, main_node_features=None, indices=None, return_attentions=False):
        if indices is not None:
            node_features = node_features[indices[0], indices[1]]

        edge_feat = self.edge_embedding.weight

        aggregated_messages = []
        for i in range(node_features.size(0)):
            message = F.relu(node_features[i] + edge_feat.mean(dim=0))
            aggregated_messages.append(message)

        aggregated_messages = torch.stack(aggregated_messages)

        updated = self.gru(aggregated_messages, node_features)

        if indices is not None:
            result = node_features.clone()
            result[indices[0], indices[1]] = updated
            return result
        return updated