"""
HECTO-E: Hierarchical Encoder with Cascaded Transformer Output and Enhanced Prototype Learning

完整模型集成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from .heterogeneous_encoder import MGHFE
from .missing_handler import MissingValueHandler, MFMHead
from .hierarchical_decoder import HAOD
from .prototype_network import DCPN
from .physics_constraints import PhysicsConstraintLayer


class HECTOE(nn.Module):
    """
    HECTO-E 完整模型

    编码器 → 缺失处理 → 异质融合 → 层次化解码 → 物理约束
                                              └→ 原型对比学习
    """

    def __init__(
        self,
        # 编码器参数
        n_num_features: int = 22,
        n_triplets: int = 6,
        n_cat_features: int = 23,
        cat_vocab_sizes: Optional[List[int]] = None,
        d_model: int = 256,
        d_emb: int = 64,
        n_heads: int = 8,
        n_transformer_layers: int = 2,
        dropout: float = 0.1,
        # 解码器参数
        n_series: int = 14,
        n_model_global: int = 64,
        specs_cls_vocab_sizes: Optional[Dict[str, int]] = None,
        n_specs_reg: int = 3,
        model_to_series: Optional[Dict[int, int]] = None,
        # 原型网络参数
        use_prototype: bool = True,
        d_proj: int = 128,
        temperature_inst: float = 0.07,
        temperature_proto: float = 0.1,
        proto_momentum: float = 0.99,
        tail_threshold: int = 10,
        # 物理约束参数
        compat_matrix: Optional[np.ndarray] = None,
    ):
        super().__init__()

        if cat_vocab_sizes is None:
            cat_vocab_sizes = [20] * 23
        if specs_cls_vocab_sizes is None:
            specs_cls_vocab_sizes = {'default': 50}

        self.d_model = d_model
        self.n_series = n_series
        self.n_model_global = n_model_global
        self.use_prototype = use_prototype

        # 缺失值处理器 (仅用于 MFM 预训练)
        # 使用与 CategoricalStream 相同的嵌入维度计算逻辑
        cat_emb_dims = []
        for vs in cat_vocab_sizes:
            if vs <= 10:
                dim = d_emb
            elif vs <= 50:
                dim = d_emb // 2
            else:
                dim = d_emb // 4
            cat_emb_dims.append(dim)

        self.missing_handler = MissingValueHandler(
            n_num_features=n_num_features,
            n_cat_features=n_cat_features,
            cat_emb_dims=cat_emb_dims,
            d_num_emb=d_model,
            d_cat_emb=d_model,
        )

        # 异质特征编码器
        self.encoder = MGHFE(
            n_num_features=n_num_features,
            n_triplets=n_triplets,
            cat_vocab_sizes=cat_vocab_sizes,
            d_model=d_model,
            d_emb=d_emb,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            dropout=dropout,
        )

        # MFM 预训练头
        self.mfm_head = MFMHead(
            d_model=d_model,
            n_num_features=n_num_features,
            n_cat_features=n_cat_features,
            cat_vocab_sizes=cat_vocab_sizes,
        )

        # 层次化解码器
        self.decoder = HAOD(
            d_model=d_model,
            n_series=n_series,
            n_model_global=n_model_global,
            specs_cls_vocab_sizes=specs_cls_vocab_sizes,
            n_specs_reg=n_specs_reg,
            n_heads=n_heads,
            dropout=dropout,
            model_to_series=model_to_series,
        )

        # 双重对比原型网络
        self.dcpn = None
        if use_prototype:
            self.dcpn = DCPN(
                n_classes=n_series,
                d_model=d_model,
                d_proj=d_proj,
                temperature_inst=temperature_inst,
                temperature_proto=temperature_proto,
                proto_momentum=proto_momentum,
                tail_threshold=tail_threshold,
            )

        # 物理约束层
        self.physics_layer = None
        if compat_matrix is not None:
            self.physics_layer = PhysicsConstraintLayer(
                compat_matrix=compat_matrix,
                model_legal_mask=self.decoder.model_legal_mask.clone(),
            )

        # 任务不确定性参数 (Homoscedastic Uncertainty)
        self.register_parameter(
            'log_sigma_l1',
            nn.Parameter(torch.zeros(1))
        )
        self.register_parameter(
            'log_sigma_l2',
            nn.Parameter(torch.zeros(1))
        )
        self.register_parameter(
            'log_sigma_l3_cls',
            nn.Parameter(torch.zeros(1))
        )
        self.register_parameter(
            'log_sigma_l3_reg',
            nn.Parameter(torch.zeros(1))
        )

    def encode(self, x_num, mask_num, x_cat, mask_cat):
        """编码器前向"""
        return self.encoder(x_num, mask_num, x_cat, mask_cat)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        mode: str = 'train',
        teacher_forcing_ratio: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        """
        x_num = batch['x_num']
        mask_num = batch['mask_num']
        x_cat = batch['x_cat']
        mask_cat = batch['mask_cat']

        B = x_num.shape[0]
        device = x_num.device

        # ── MFM 预训练 ──
        if mode == 'pretrain':
            # 使用 MissingHandler 处理缺失特征
            h_num, h_cat = self.missing_handler(
                x_num, mask_num, x_cat, mask_cat,
                cat_embeddings=self.encoder.cat_stream.embeddings
            )
            h_num_pool = h_num.mean(dim=1)
            h_cat_pool = h_cat.mean(dim=1)
            h_fused, gate_weights = self.encoder.gate_fusion(
                h_num_pool, h_cat_pool, mask_num, mask_cat
            )
            pred_num, pred_cat = self.mfm_head(h_fused)
            return {
                'h_fused': h_fused,
                'gate_weights': gate_weights,
                'mfm_pred_num': pred_num,
                'mfm_pred_cat': pred_cat,
                'mfm_targets_num': batch.get('masked_targets_num'),
                'mfm_targets_cat': batch.get('masked_targets_cat'),
            }

        # ── 正常前向 ──
        enc_out = self.encoder(x_num, mask_num, x_cat, mask_cat)
        h_fused = enc_out['h_fused']
        gate_weights = enc_out['gate_weights']

        outputs = {
            'h_fused': h_fused,
            'gate_weights': gate_weights,
        }

        # 解码
        y_series = batch.get('y_series')
        y_model = batch.get('y_model')
        dec_out = self.decoder(
            h_fused,
            y_series=y_series,
            y_model=y_model,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        outputs.update(dec_out)

        # 原型网络
        if self.dcpn is not None and y_series is not None:
            z = self.dcpn(h_fused)
            outputs['z_contrastive'] = z

        return outputs

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        mode: str = 'train',
        contrastive_weight: float = 0.1,
        proto_weight: float = 0.05,
        physics_weight: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有损失

        Returns:
            {'L_total': ..., 'L_series': ..., 'L_model': ..., ...}
        """
        losses = {}
        device = outputs['h_fused'].device

        # MFM 预训练损失
        if mode == 'pretrain':
            mfm_num_loss = F.mse_loss(
                outputs['mfm_pred_num'],
                outputs['mfm_targets_num'],
                reduction='none'
            )
            # 仅在被遮蔽位置计算损失
            mask_mfm = (outputs['mfm_targets_num'] != 0).float()
            losses['L_mfm_num'] = (mfm_num_loss * mask_mfm).sum() / (mask_mfm.sum() + 1e-8) * 0.1

            if 'mfm_targets_cat' in outputs and outputs['mfm_targets_cat'] is not None:
                mfm_cat_loss = torch.tensor(0.0, device=device)
                for i, cat_logits in enumerate(outputs['mfm_pred_cat']):
                    targets = outputs['mfm_targets_cat'][:, i].long()
                    mask_c = (targets != 0)
                    if mask_c.sum() > 0:
                        mfm_cat_loss += F.cross_entropy(
                            cat_logits[mask_c], targets[mask_c]
                        )
                losses['L_mfm_cat'] = mfm_cat_loss * 0.1

            losses['L_total'] = losses.get('L_mfm_num', 0.0) + losses.get('L_mfm_cat', 0.0)
            return losses

        # ── 层次化分类损失 ──
        y_series = batch['y_series']
        y_model = batch['y_model']

        # Level 1: 产品系列 — Focal Loss
        losses['L_series'] = self._focal_loss(
            outputs['logits_series'], y_series, gamma=2.0
        )

        # Level 2: 产品型号 — Focal Loss
        losses['L_model'] = self._focal_loss(
            outputs['logits_model'], y_model, gamma=2.0
        )

        # Level 3: 分类损失
        l3_cls_total = torch.tensor(0.0, device=device)
        n_l3_cls = 0
        for name, logits in outputs['logits_specs_cls'].items():
            # 从 batch 中提取对应的标签
            batch_key = f'y_specs_cls_{name}'
            if batch_key in batch:
                labels = batch[batch_key].long()
                valid_mask = labels >= 0
                if valid_mask.sum() > 0:
                    l3_cls_total += F.cross_entropy(
                        logits[valid_mask], labels[valid_mask],
                        reduction='mean'
                    )
                    n_l3_cls += 1
        losses['L_specs_cls'] = l3_cls_total / max(n_l3_cls, 1)

        # Level 3: 回归损失 — Huber Loss
        l3_reg_total = torch.tensor(0.0, device=device)
        n_l3_reg = 0
        for name, pred in outputs['preds_specs_reg'].items():
            batch_key = f'y_specs_reg_{name}'
            if batch_key in batch:
                targets = batch[batch_key]
                valid_mask = targets != 0  # 0 = missing
                if valid_mask.sum() > 0:
                    l3_reg_total += F.huber_loss(
                        pred[valid_mask], targets[valid_mask],
                        delta=1.0, reduction='mean'
                    )
                    n_l3_reg += 1
        losses['L_specs_reg'] = l3_reg_total / max(n_l3_reg, 1)

        # ── 对比学习损失 ──
        if self.dcpn is not None and 'z_contrastive' in outputs:
            z = outputs['z_contrastive']
            losses['L_inst'] = self.dcpn.compute_instance_loss(z, y_series)
            losses['L_proto'] = self.dcpn.compute_prototype_loss(z, y_series, use_aug=True)

            # 动量更新原型（训练模式）
            if mode == 'train':
                self.dcpn.update_prototypes(z.detach(), y_series)
        else:
            losses['L_inst'] = torch.tensor(0.0, device=device)
            losses['L_proto'] = torch.tensor(0.0, device=device)

        # ── 物理约束损失 ──
        if self.physics_layer is not None:
            labels_for_physics = {'specs_cls': {}}
            for safe_name in outputs['logits_specs_cls'].keys():
                # 尝试从 batch 中找到匹配的标签
                for batch_key, batch_val in batch.items():
                    if batch_key.startswith('y_specs_cls_'):
                        orig_name = batch_key[len('y_specs_cls_'):]
                        orig_safe = orig_name.replace('.', '_').replace('/', '_')
                        if orig_safe == safe_name:
                            labels_for_physics['specs_cls'][safe_name] = batch_val
                            break
            # 至少填充 body/core 标签
            body_safe = '阀体材质_1'
            core_safe = '阀芯材质_1'
            if body_safe not in labels_for_physics['specs_cls']:
                labels_for_physics['specs_cls'][body_safe] = torch.zeros(B, dtype=torch.long, device=device)
            if core_safe not in labels_for_physics['specs_cls']:
                labels_for_physics['specs_cls'][core_safe] = torch.zeros(B, dtype=torch.long, device=device)

            physics_losses = self.physics_layer(outputs, labels_for_physics)
            losses.update(physics_losses)
        else:
            losses['L_physics'] = torch.tensor(0.0, device=device)

        # ── 不确定性加权损失 ──
        # L_w = L / (2*sigma^2) + log(sigma)
        w1 = torch.exp(-2 * self.log_sigma_l1)
        w2 = torch.exp(-2 * self.log_sigma_l2)
        w3c = torch.exp(-2 * self.log_sigma_l3_cls)
        w3r = torch.exp(-2 * self.log_sigma_l3_reg)

        L_task = (
            w1 * losses['L_series'] + self.log_sigma_l1.squeeze() +
            w2 * losses['L_model'] + self.log_sigma_l2.squeeze() +
            w3c * losses['L_specs_cls'] + self.log_sigma_l3_cls.squeeze() +
            w3r * losses['L_specs_reg'] + self.log_sigma_l3_reg.squeeze()
        )

        losses['L_total'] = (
            L_task +
            contrastive_weight * losses['L_inst'] +
            proto_weight * losses['L_proto'] +
            physics_weight * losses['L_physics']
        )

        return losses

    def _focal_loss(self, logits, targets, gamma=2.0, alpha=None):
        """Focal Loss for imbalanced classification"""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** gamma
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha_t = torch.tensor(alpha, device=logits.device)[targets]
            else:
                alpha_t = alpha
            focal_weight = alpha_t * focal_weight
        return (focal_weight * ce_loss).mean()

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """推理模式"""
        self.eval()
        return self.forward(batch, mode='eval', teacher_forcing_ratio=0.0)
