"""
层次化自回归输出解码器 (HAOD — Hierarchical Autoregressive Output Decoder)

三层级联：产品系列(Level 1) → 产品型号(Level 2) → 详细规格(Level 3)
每层使用 Cross-Attention 查询编码器特征，前一层预测作为当前层的条件。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class LevelDecoder(nn.Module):
    """单个层级的解码模块"""

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """
        query:      [B, d_model]  (层级查询向量)
        key_value:  [B, 1, d_model] 或 [B, d_model]  (编码器融合特征)
        """
        if query.dim() == 2:
            query = query.unsqueeze(1)     # [B, 1, d_model]
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)

        h, _ = self.cross_attn(query, key_value, key_value)
        h = self.norm1(query + self.dropout(h))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h.squeeze(1)


class HAOD(nn.Module):
    """
    层次化自回归输出解码器

    三层级联预测:
      Level 1: 产品系列 (14 类)
      Level 2: 产品型号 (64 类，按系列条件化)
      Level 3: 详细规格 (17 分类 + 3 回归)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_series: int = 14,
        n_model_global: int = 64,
        model_vocab_by_series: Optional[Dict[int, int]] = None,
        specs_cls_vocab_sizes: Optional[Dict[str, int]] = None,
        n_specs_reg: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        model_to_series: Optional[Dict[int, int]] = None,
    ):
        """
        Args:
            model_vocab_by_series: {series_id: n_models_in_series}
            specs_cls_vocab_sizes: {'col_name': vocab_size}
            model_to_series: {model_global_id: series_id}  合法性映射
        """
        super().__init__()
        self.d_model = d_model
        self.n_series = n_series
        self.n_model_global = n_model_global
        self.n_specs_reg = n_specs_reg
        self.model_to_series = model_to_series or {}

        # 层级嵌入：告诉解码器当前在预测哪个层级
        self.level_embeddings = nn.Parameter(torch.randn(3, d_model) * 0.02)

        # 起始标记
        self.start_token = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # 前一层预测的嵌入
        self.series_out_emb = nn.Embedding(n_series, d_model)
        self.model_out_emb = nn.Embedding(n_model_global, d_model)

        # 三层解码器
        self.decoder_l1 = LevelDecoder(d_model, n_heads, dropout)
        self.decoder_l2 = LevelDecoder(d_model, n_heads, dropout)
        self.decoder_l3 = LevelDecoder(d_model, n_heads, dropout)

        # 输出头
        self.head_series = nn.Linear(d_model, n_series)
        self.head_model = nn.Linear(d_model, n_model_global)
        self.head_specs_cls = nn.ModuleDict()
        self.head_specs_reg = nn.ModuleDict()

        self.specs_cls_keys = []
        if specs_cls_vocab_sizes:
            for name, vs in specs_cls_vocab_sizes.items():
                safe_name = name.replace('.', '_').replace('/', '_')
                self.head_specs_cls[safe_name] = nn.Linear(d_model, vs)
                self.specs_cls_keys.append((safe_name, vs))

        for i in range(n_specs_reg):
            self.head_specs_reg[f'reg_{i}'] = nn.Linear(d_model, 1)

        self.specs_reg_keys = [f'reg_{i}' for i in range(n_specs_reg)]

        # 合法性掩码 [n_series, n_model_global]: series s → models under s
        self.register_buffer(
            'model_legal_mask',
            self._build_model_legal_mask()
        )

        self.dropout = nn.Dropout(dropout)

    def _build_model_legal_mask(self) -> torch.Tensor:
        """构建系列→型号合法性掩码"""
        mask = torch.zeros(self.n_series, self.n_model_global)
        for model_id, series_id in self.model_to_series.items():
            if 0 <= series_id < self.n_series and 0 <= model_id < self.n_model_global:
                mask[series_id, model_id] = 1.0
        if mask.sum() == 0:
            mask = torch.ones(self.n_series, self.n_model_global)
        return mask

    def forward(
        self,
        h_fused: torch.Tensor,
        y_series: Optional[torch.Tensor] = None,
        y_model: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
        return_all: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h_fused:  编码器融合特征 [B, d_model]
            y_series: 真实产品系列 [B]（训练时）
            y_model:  真实产品型号 [B]（训练时）
            teacher_forcing_ratio: 教师强制比例

        Returns:
            logits_series, logits_model, logits_specs_cls, preds_specs_reg,
            pred_series, pred_model
        """
        B = h_fused.shape[0]
        device = h_fused.device

        # ── Level 1: 产品系列 ──
        q1 = h_fused + self.start_token + self.level_embeddings[0]
        h1 = self.decoder_l1(q1, h_fused)
        logits_series = self.head_series(h1)  # [B, n_series]
        pred_series = logits_series.argmax(dim=-1)

        # 决定 Level 1 → Level 2 使用的嵌入
        use_teacher_l1 = (torch.rand(1, device=device).item() < teacher_forcing_ratio) and (y_series is not None)
        series_emb_input = self.series_out_emb(y_series if use_teacher_l1 else pred_series)

        # ── Level 2: 产品型号（系列条件化）──
        q2 = h_fused + series_emb_input + self.level_embeddings[1]
        h2 = self.decoder_l2(q2, h_fused)
        logits_model_raw = self.head_model(h2)  # [B, n_model_global]

        # 系列条件化掩码：仅允许该系列下的合法型号
        series_for_mask = y_series if use_teacher_l1 else pred_series
        model_mask = self.model_legal_mask[series_for_mask]  # [B, n_model_global]
        logits_model = logits_model_raw.masked_fill(model_mask < 0.5, float('-inf'))

        pred_model = logits_model.argmax(dim=-1)

        # 决定 Level 2 → Level 3 使用的嵌入
        use_teacher_l2 = (torch.rand(1, device=device).item() < teacher_forcing_ratio) and (y_model is not None)
        model_emb_input = self.model_out_emb(y_model if use_teacher_l2 else pred_model)

        # ── Level 3: 详细规格 ──
        q3 = h_fused + series_emb_input + model_emb_input + self.level_embeddings[2]
        h3 = self.decoder_l3(q3, h_fused)

        logits_specs_cls = {}
        for name, head in self.head_specs_cls.items():
            logits_specs_cls[name] = head(h3)

        preds_specs_reg = {}
        for name, head in self.head_specs_reg.items():
            preds_specs_reg[name] = head(h3).squeeze(-1)

        out = {
            'logits_series': logits_series,
            'logits_model': logits_model,
            'logits_specs_cls': logits_specs_cls,
            'preds_specs_reg': preds_specs_reg,
            'pred_series': pred_series,
            'pred_model': pred_model,
            'h_l1': h1,
            'h_l2': h2,
            'h_l3': h3,
        }
        return out

    def predict(self, h_fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        """推理模式：自回归（不使用 Teacher Forcing）"""
        return self.forward(h_fused, teacher_forcing_ratio=0.0, return_all=True)

    def get_hierarchical_consistency_loss(
        self, logits_model: torch.Tensor, pred_series: torch.Tensor
    ) -> torch.Tensor:
        """层级一致性损失：预测的型号必须在预测的系列下合法"""
        model_mask = self.model_legal_mask[pred_series]
        # 计算模型 logits 在非法区域的平均值，惩罚高值
        illegal_logits = logits_model * (1 - model_mask)
        illegal_penalty = illegal_logits.clamp(min=0).mean()
        return illegal_penalty * 0.1
