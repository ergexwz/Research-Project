"""
缺失数据处理与 MFM 预训练模块

每个特征位置有专属的可学习缺失嵌入向量。
MFM (Masked Feature Modeling) 预训练让编码器学会推断缺失特征。
"""

import torch
import torch.nn as nn


class MissingValueHandler(nn.Module):
    """
    缺失值处理器

    对每个数值和类别特征位置维护独立的缺失嵌入向量。
    输入: x_num [B, N_num], mask_num [B, N_num], x_cat [B, N_cat], mask_cat [B, N_cat]
    输出: h_num [B, N_num, d_num], h_cat [B, N_cat, d_cat]
    """

    def __init__(
        self,
        n_num_features: int,
        n_cat_features: int,
        cat_emb_dims: list = None,
        d_num_emb: int = 256,
        d_cat_emb: int = 256,
    ):
        super().__init__()
        self.num_missing_emb = nn.Parameter(torch.randn(n_num_features, d_num_emb) * 0.02)
        self.num_proj = nn.Linear(1, d_num_emb)

        # 类别: 如果外部嵌入维度不同，需要对齐投影
        self.cat_emb_dims = cat_emb_dims or []
        self.cat_proj = nn.ModuleList()
        for dim in self.cat_emb_dims:
            self.cat_proj.append(
                nn.Linear(dim, d_cat_emb) if dim != d_cat_emb else nn.Identity()
            )

        self.cat_missing_emb = nn.Parameter(torch.randn(n_cat_features, d_cat_emb) * 0.02)

        self.n_num_features = n_num_features
        self.n_cat_features = n_cat_features
        self.d_num_emb = d_num_emb
        self.d_cat_emb = d_cat_emb

    def forward(self, x_num, mask_num, x_cat, mask_cat, cat_embeddings=None):
        """
        Args:
            x_num:      [B, N_num] 数值特征（已标准化）
            mask_num:   [B, N_num] 1=已知, 0=缺失
            x_cat:      [B, N_cat] 类别索引
            mask_cat:   [B, N_cat] 1=已知, 0=缺失
            cat_embeddings: list of nn.Embedding | None, 外部传入的类别嵌入表

        Returns:
            h_num:  [B, N_num, d_num_emb]
            h_cat:  [B, N_cat, d_cat]
        """
        B = x_num.shape[0]

        # 数值: 标量→嵌入 + 位置编码（缺失嵌入）
        # [B, N_num] → [B, N_num, 1] → [B, N_num, d_num_emb]
        x_num_expanded = x_num.unsqueeze(-1)
        h_num_obs = self.num_proj(x_num_expanded)

        # 使用缺失嵌入替代缺失位置
        miss_emb_num = self.num_missing_emb.unsqueeze(0).expand(B, -1, -1)
        mask_num_3d = mask_num.unsqueeze(-1)
        h_num = mask_num_3d * h_num_obs + (1 - mask_num_3d) * miss_emb_num

        # 类别: 查嵌入表并使用对齐投影
        if cat_embeddings is not None and len(self.cat_proj) > 0:
            h_cat = torch.zeros(B, self.n_cat_features, self.d_cat_emb,
                                device=x_num.device, dtype=x_num.dtype)
            for i, emb_layer in enumerate(cat_embeddings):
                if emb_layer is not None and i < len(self.cat_proj):
                    raw_emb = emb_layer(x_cat[:, i])  # [B, native_dim]
                    h_cat[:, i] = self.cat_proj[i](raw_emb)  # [B, d_cat_emb]
        else:
            h_cat_obs = self.num_proj(x_cat.unsqueeze(-1).float())
            h_cat = h_cat_obs

        # 缺失嵌入替换
        miss_emb_cat = self.cat_missing_emb.unsqueeze(0).expand(B, -1, -1)
        mask_cat_3d = mask_cat.unsqueeze(-1)
        h_cat = mask_cat_3d * h_cat + (1 - mask_cat_3d) * miss_emb_cat

        return h_num, h_cat


class MFMHead(nn.Module):
    """
    缺失特征建模预训练头

    从编码器的融合表示中重建被遮蔽的特征值。
    数值用 MSE，类别用 CrossEntropy。
    """

    def __init__(
        self,
        d_model: int,
        n_num_features: int,
        n_cat_features: int,
        cat_vocab_sizes: list,
    ):
        super().__init__()
        self.num_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, n_num_features),
        )
        self.cat_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Linear(64, vs),
            ) if vs > 0 else nn.Identity()
            for vs in cat_vocab_sizes
        ])

    def forward(self, h_fused):
        """h_fused: [B, d_model]"""
        pred_num = self.num_head(h_fused)
        pred_cat = []
        for head in self.cat_heads:
            pred_cat.append(head(h_fused))
        return pred_num, pred_cat
