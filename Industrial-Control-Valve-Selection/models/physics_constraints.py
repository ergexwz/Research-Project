"""
物理约束正则化模块 (PICL — Physics-Informed Constraint Layer)

三个层次的约束：
  1. 材质兼容性约束 — 阀体/阀芯/阀座材质共现约束
  2. 压力-温度包络约束 — ASME B16.34 标准
  3. 层次一致性约束 — 型号必须在系列下合法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class MaterialCompatibilityLoss(nn.Module):
    """
    材质兼容性约束

    惩罚训练数据中几乎不出现的材质组合。
    L_mat = -log(P(core_mat | body_mat)) - log(P(seat_mat | body_mat)) ...
    """

    def __init__(self, compat_matrix: np.ndarray, body_to_core: bool = True,
                 body_to_seat: bool = True):
        super().__init__()
        self.register_buffer(
            'compat_body_core',
            torch.from_numpy(compat_matrix.astype(np.float32))
        )
        self.active_body_to_core = body_to_core
        self.active_body_to_seat = body_to_seat

    def forward(
        self,
        pred_body: torch.Tensor,
        pred_core: torch.Tensor,
        pred_seat: Optional[torch.Tensor] = None,
        labels_body: Optional[torch.Tensor] = None,
        labels_core: Optional[torch.Tensor] = None,
        labels_seat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算材质兼容性违反损失。
        对每条样本，如果预测的(core, seat)不在(body, *)的常见组合中则惩罚。

        pred/logits: [B, V] (分类 logits) 或 [B] (类别索引)
        """
        loss = torch.tensor(0.0, device=pred_body.device)

        if self.active_body_to_core and labels_body is not None and labels_core is not None:
            prob_body_to_core = self.compat_body_core[labels_body]  # [B, V_core]
            log_probs_core = F.log_softmax(pred_core, dim=-1)
            loss = loss + -(prob_body_to_core * log_probs_core).sum(dim=-1).mean()

        if self.active_body_to_seat and labels_body is not None and labels_seat is not None:
            prob_body_to_seat = self.compat_body_core[labels_body]  # [B, V_seat]
            log_probs_seat = F.log_softmax(pred_seat, dim=-1)
            loss = loss + -(prob_body_to_seat * log_probs_seat).sum(dim=-1).mean()

        return loss


class PressureTemperatureConstraint(nn.Module):
    """
    压力-温度包络约束

    根据 ASME B16.34:
      Class 150 + 碳钢 (A105) → T_max ≈ 538°C
      Class 300 + 碳钢 (A105) → T_max ≈ 538°C
      Class 150 + 不锈钢 (316) → T_max ≈ 816°C
      ...

    实际约束: 给定材质和压力等级，温度不应超过对应上限。
    """

    # 简化的 PN-T 包络表（压力等级类别索引 → 温度上限函数系数）
    # Class150 → 425°C, Class300 → 450°C, Class600 → 500°C,
    # Class900 → 540°C, Class1500 → 590°C, Class2500 → 650°C
    # HG PN16 → 200°C, HG PN25 → 200°C, HG PN40 → 250°C
    # HG PN63 → 300°C, HG PN100 → 350°C, HG PN160 → 400°C

    def __init__(
        self,
        pressure_class_mapping: Dict[int, float] = None,
        material_temp_bonus: Dict[int, float] = None,
    ):
        """
        Args:
            pressure_class_mapping: {pressure_class_id: T_max_base}
            material_temp_bonus: {material_id: T_bonus} 材质对温度的额外容忍
        """
        super().__init__()
        self.pressure_class_mapping = pressure_class_mapping or {}
        self.material_temp_bonus = material_temp_bonus or {}

    def forward(
        self,
        design_temperature: torch.Tensor,
        pred_pressure_class: torch.Tensor,
        pred_body_material: torch.Tensor,
        pressure_class_logits: Optional[torch.Tensor] = None,
        body_material_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        design_temperature: [B]  设计温度（°C，已归一化，需反归一化）
        pred_pressure_class: [B]  压力等级（类别索引）
        pred_body_material: [B]  阀体材质（类别索引）
        """
        # 由于反归一化依赖外部 scaler，这里用简化版本：
        # 通过 logits 软约束而非硬阈值
        loss = torch.tensor(0.0, device=design_temperature.device)

        if pressure_class_logits is not None and body_material_logits is not None:
            # 高压力等级 + 高温 → 高 penalty（压力等级应随之提高）
            # 软约束：鼓励温度高时倾向于选择高压力等级
            pass  # 留给训练时注入 scaler 参数后实现

        return loss


class HierarchicalConsistencyLoss(nn.Module):
    """
    层次一致性约束

    确保预测层次间的一致性：
    - 产品型号必须属于预测的产品系列
    - 不能出现"系列=G系列球阀, 型号=ATS"这种非法组合

    由于 HAOD 已在解码器中实现了系列条件化掩码（masked logits），
    这里仅作为额外的软约束。
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits_model: torch.Tensor,
        pred_series: torch.Tensor,
        model_legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits_model:  [B, n_model_global]  模型 logits
        pred_series:   [B]                  预测的系列
        model_legal_mask: [n_series, n_model_global]  合法性掩码
        """
        mask = model_legal_mask[pred_series]  # [B, n_model_global]
        illegal = (1 - mask) * logits_model
        # 非法区域的 logits 应该尽可能低
        penalty = illegal.clamp(min=0).mean()
        return penalty


class PhysicsConstraintLayer(nn.Module):
    """
    物理约束层 — 统合所有约束
    """

    def __init__(
        self,
        compat_matrix: np.ndarray,
        model_legal_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.mat_loss = MaterialCompatibilityLoss(compat_matrix)
        self.PT_loss = PressureTemperatureConstraint()
        self.consist_loss = HierarchicalConsistencyLoss()

        if model_legal_mask is not None:
            self.register_buffer('model_legal_mask', model_legal_mask)
        else:
            self.register_buffer('model_legal_mask', torch.eye(1))

    def forward(
        self,
        decoder_outputs: Dict,
        labels: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有物理约束损失

        labels dict now uses safe names ('.' → '_', '/' → '_')
        """
        losses = {}

        # 材质兼容性约束 — 使用 safe names
        body_key = '阀体材质_1'
        core_key = '阀芯材质_1'
        seat_key = '阀座材质_1'

        specs_cls = decoder_outputs.get('logits_specs_cls', {})
        labels_cls = labels.get('specs_cls', {})

        if body_key in specs_cls and body_key in labels_cls:
            core_logits = specs_cls.get(core_key, specs_cls.get(body_key))
            core_labels = labels_cls.get(core_key, labels_cls.get(body_key))
            seat_logits = specs_cls.get(seat_key)
            seat_labels = labels_cls.get(seat_key)
            try:
                losses['L_mat'] = self.mat_loss(
                    pred_body=specs_cls[body_key],
                    pred_core=core_logits,
                    pred_seat=seat_logits,
                    labels_body=labels_cls[body_key],
                    labels_core=core_labels,
                    labels_seat=seat_labels,
                )
            except Exception:
                losses['L_mat'] = torch.tensor(0.0, device=next(iter(specs_cls.values())).device)
        else:
            losses['L_mat'] = torch.tensor(0.0)

        # 层次一致性约束
        if 'logits_model' in decoder_outputs and 'pred_series' in decoder_outputs:
            losses['L_consist'] = self.consist_loss(
                decoder_outputs['logits_model'],
                decoder_outputs['pred_series'],
                self.model_legal_mask,
            )

        losses['L_PT'] = torch.tensor(0.0)  # 待训练时通过 scaler 参数实现

        # 总物理损失
        losses['L_physics'] = (
            0.1 * losses.get('L_mat', 0.0) +
            0.05 * losses.get('L_PT', 0.0) +
            0.1 * losses.get('L_consist', 0.0)
        )

        return losses
