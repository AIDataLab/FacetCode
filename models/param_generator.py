import torch
from torch import nn
import torch.nn.functional as F
from models.mlp import MLP


class ICDParameterGenerator(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, pos_enc_dim=64, lora_rank=2, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pos_enc_dim = pos_enc_dim
        self.lora_rank = lora_rank
        self.num_layers = num_layers

        self.code_encoder = nn.Linear(input_dim, hidden_dim)

        self.layer_pos_embeddings = nn.Parameter(torch.randn(num_layers, pos_enc_dim))

        self.down_proj = MLP(hidden_dim + pos_enc_dim, hidden_dim, lora_rank, num_layers=1, act='gelu')
        self.up_proj = MLP(hidden_dim + pos_enc_dim, hidden_dim, input_dim * lora_rank, num_layers=1, act='gelu')

        self.out_proj_B = nn.Linear(lora_rank, input_dim, bias=False)
        self.out_proj_A = nn.Linear(input_dim, lora_rank, bias=False)

    def forward(self, code_description, layer_idx):
        h = self.code_encoder(code_description)

        pos_emb = self.layer_pos_embeddings[layer_idx]

        h_pos = torch.cat([h, pos_emb], dim=-1)

        B_raw = self.down_proj(h_pos)
        A_raw = self.up_proj(h_pos)

        B = self.out_proj_B(B_raw)
        A = self.out_proj_A(A_raw)

        return A, B

    def generate_full_adapter(self, code_description, num_layers):
        adapters_A = []
        adapters_B = []
        for layer_idx in range(num_layers):
            A, B = self.forward(code_description, layer_idx)
            adapters_A.append(A)
            adapters_B.append(B)
        return adapters_A, adapters_B


class ParameterGeneratorLoss(nn.Module):
    def __init__(self, lambda_dist=0.01):
        super().__init__()
        self.lambda_dist = lambda_dist

    def forward(self, pred_adapters_A, pred_adapters_B, target_adapters_A, target_adapters_B,
                base_model=None, code_description=None):
        struct_loss = 0.0
        for pred_A, pred_B, tgt_A, tgt_B in zip(pred_adapters_A, pred_adapters_B,
                                                 target_adapters_A, target_adapters_B):
            struct_loss += F.mse_loss(pred_A, tgt_A) + F.mse_loss(pred_B, tgt_B)

        kl_loss = 0.0
        if base_model is not None and code_description is not None:
            with torch.no_grad():
                target_logits = base_model(code_description, adapter=True)
            pred_logits = base_model(code_description, adapter=False)
            kl_loss = F.kl_div(F.log_softmax(pred_logits, dim=-1),
                               F.softmax(target_logits, dim=-1), reduction='batchmean')

        total_loss = struct_loss + self.lambda_dist * kl_loss
        return total_loss, struct_loss, kl_loss