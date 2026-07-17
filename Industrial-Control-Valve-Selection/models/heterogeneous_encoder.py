"""
多粒度异质特征编码器 (MGHFE — Multi-Granularity Heterogeneous Feature Encoder)

核心创新：交叉模态门控融合，根据样本的模态完整性自适应调节数值/类别特征的贡献。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(1), :].unsqueeze(0))


class TripletConvEncoder(nn.Module):
    """
    三元组卷积编码器

    每个物理量的 (max, nor, min) 三元组构成工况轮廓
    使用 1D-CNN 提取轮廓形态特征
    """

    def __init__(self, n_triplets: int = 6, d_out: int = 64):
        super().__init__()
        self.n_triplets = n_triplets
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(32 * 3, d_out)

    def forward(self, x):
        """
        x: [B, 18] — 6个三元组 * 3
        返回: [B, n_triplets, d_out]
        """
        B = x.shape[0]
        x = x.view(B, self.n_triplets, 3)      # [B, 6, 3]
        x = x.reshape(B * self.n_triplets, 1, 3)  # [B*6, 1, 3]
        h = self.conv(x)                         # [B*6, 32, 3]
        h = h.reshape(B, self.n_triplets, -1)    # [B, 6, 96]
        h = self.proj(h)                         # [B, 6, d_out]
        return h


class NumericalStream(nn.Module):
    """
    数值特征流

    数值特征 → 三元组CNN + 独立数值MLP → 拼接投影 → d_model
    """

    def __init__(
        self,
        n_num_features: int = 22,
        n_triplets: int = 6,
        d_triplet: int = 64,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.triplet_encoder = TripletConvEncoder(n_triplets, d_triplet)
        self.n_triplets = n_triplets

        # 独立数值特征 MLP（18 个三元组之外的 4 个独立特征）
        n_independent = n_num_features - n_triplets * 3
        assert n_independent >= 0, f"n_num_features {n_num_features} < n_triplets*3 {n_triplets*3}"

        self.independent_proj = nn.Sequential(
            nn.Linear(n_independent, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_triplet * n_triplets),
        )

        # 拼接投影
        self.proj = nn.Sequential(
            nn.Linear(d_triplet * n_triplets * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
        )

        self.n_triplets = n_triplets
        self.n_independent = n_independent

    def forward(self, x_num):
        """
        x_num: [B, 22] — 前 18 个是三元组，后 4 个是独立特征
        """
        B = x_num.shape[0]
        triplets = x_num[:, :self.n_triplets * 3]
        independent = x_num[:, self.n_triplets * 3:]

        h_triplet = self.triplet_encoder(triplets)          # [B, 6, 64]
        h_triplet_flat = h_triplet.reshape(B, -1)            # [B, 384]

        h_independent = self.independent_proj(independent)   # [B, 384]

        h_concat = torch.cat([h_triplet_flat, h_independent], dim=-1)  # [B, 768]
        h_num = self.proj(h_concat)                          # [B, d_model]
        return h_num


class CategoricalStream(nn.Module):
    """
    类别特征流

    类别特征 → Entity Embedding → TabTransformer Self-Attention → 池化 → d_model
    """

    def __init__(
        self,
        vocab_sizes: List[int],
        d_emb: int = 64,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = len(vocab_sizes)

        # 动态嵌入维度：高基数小维度，低基数大维度
        self.embeddings = nn.ModuleList()
        self.emb_dims = []
        for vs in vocab_sizes:
            if vs <= 10:
                dim = d_emb
            elif vs <= 50:
                dim = d_emb // 2
            else:
                dim = d_emb // 4
            self.embeddings.append(nn.Embedding(vs, dim, padding_idx=None))
            self.emb_dims.append(dim)

        # 维度对齐投影
        total_emb_dim = sum(self.emb_dims)
        self.dim_align = nn.Linear(total_emb_dim, d_model)

        self.pos_encoding = PositionalEncoding(d_model, max_len=64)

        # TabTransformer: Multi-head Self-Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_cat, mask_cat):
        """
        x_cat:    [B, N_cat]
        mask_cat: [B, N_cat]
        """
        B = x_cat.shape[0]
        embs = []
        for i, emb_layer in enumerate(self.embeddings):
            embs.append(emb_layer(x_cat[:, i]))
        h = torch.cat(embs, dim=-1)                    # [B, total_emb_dim]
        h = self.dim_align(h)                           # [B, d_model]
        h = h.unsqueeze(1)                              # [B, 1, d_model]
        h = self.pos_encoding(h)
        h = self.transformer(h)                         # [B, 1, d_model]

        # 全局池化（单个 token 时等价于 squeeze）
        h_mean = h.mean(dim=1)                          # [B, d_model]
        h_max = h.max(dim=1).values                     # [B, d_model]
        h_cat = h_mean + h_max                          # [B, d_model]
        return h_cat


class CrossModalGateFusion(nn.Module):
    """
    交叉模态门控融合

    核心创新：根据每个样本的模态完整性，自适应调节数值流和类别流的贡献。

    gate_num = σ(W_gate_num @ [h_num; h_cat; missing_rate_num; missing_rate_cat])
    gate_cat = σ(W_gate_cat @ [h_num; h_cat; missing_rate_num; missing_rate_cat])
    h_fused = gate_num ⊙ h_num + gate_cat ⊙ h_cat + gate_bias
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        input_dim = d_model * 2 + 2  # h_num + h_cat + 2 missing rates
        self.W_gate_num = nn.Linear(input_dim, d_model)
        self.W_gate_cat = nn.Linear(input_dim, d_model)
        self.W_bias = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, h_num, h_cat, mask_num, mask_cat):
        """
        h_num: [B, d_model]
        h_cat: [B, d_model]
        mask_num: [B, N_num]
        mask_cat: [B, N_cat]
        """
        missing_rate_num = (1 - mask_num).mean(dim=1, keepdim=True)
        missing_rate_cat = (1 - mask_cat).mean(dim=1, keepdim=True)

        gate_input = torch.cat([h_num, h_cat, missing_rate_num, missing_rate_cat], dim=-1)

        gate_num = torch.sigmoid(self.W_gate_num(gate_input))
        gate_cat = torch.sigmoid(self.W_gate_cat(gate_input))
        gate_bias = self.W_bias(gate_input)

        h_fused = gate_num * h_num + gate_cat * h_cat + gate_bias
        h_fused = self.layer_norm(h_fused)
        return h_fused, (gate_num, gate_cat)


class MGHFE(nn.Module):
    """
    多粒度异质特征编码器

    组合 NumericalStream + CategoricalStream + CrossModalGateFusion
    """

    def __init__(
        self,
        n_num_features: int = 22,
        n_triplets: int = 6,
        cat_vocab_sizes: List[int] = None,
        d_model: int = 256,
        d_emb: int = 64,
        n_heads: int = 8,
        n_transformer_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if cat_vocab_sizes is None:
            cat_vocab_sizes = [20] * 23  # 默认 23 个类别特征

        self.num_stream = NumericalStream(
            n_num_features=n_num_features,
            n_triplets=n_triplets,
            d_model=d_model,
            dropout=dropout,
        )
        self.cat_stream = CategoricalStream(
            vocab_sizes=cat_vocab_sizes,
            d_emb=d_emb,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_transformer_layers,
            dropout=dropout,
        )
        self.gate_fusion = CrossModalGateFusion(d_model)
        self.d_model = d_model

    def forward(self, x_num, mask_num, x_cat, mask_cat):
        h_num = self.num_stream(x_num)
        h_cat = self.cat_stream(x_cat, mask_cat)
        h_fused, gate_weights = self.gate_fusion(h_num, h_cat, mask_num, mask_cat)
        return {
            'h_fused': h_fused,
            'h_num': h_num,
            'h_cat': h_cat,
            'gate_weights': gate_weights,
        }
