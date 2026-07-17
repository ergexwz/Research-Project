"""
双重对比原型网络 (DCPN — Dual Contrastive Prototype Network)

两个层次的对比学习：
  1. 实例级监督对比 (Supervised Contrastive Loss / SupCon)
  2. 类原型对比 (Prototype-based Contrastive with Momentum Update)

针对长尾分布：原型插值增强策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ProjectionHead(nn.Module):
    """对比学习投影头"""

    def __init__(self, d_in: int = 256, d_hidden: int = 128, d_out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class SupConLoss(nn.Module):
    """
    监督对比损失 (Supervised Contrastive Loss)

    正样本：同类别（产品系列相同）的所有样本
    负样本：不同类别的样本
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: [B, d]  L2-normalized
        labels:   [B]
        """
        B = features.shape[0]
        device = features.device

        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 相似度矩阵
        sim = features @ features.T / self.tau  # [B, B]

        # 正样本掩码：同标签
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [B, B]
        pos_mask.fill_diagonal_(0.0)  # 去除自身

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 计算 InfoNCE
        exp_sim = torch.exp(sim)
        exp_sim = exp_sim - exp_sim.diag().diag()  # 去除自身

        # log-sum-exp
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # 每样本的正样本对数概率均值
        pos_per_sample = pos_mask.sum(dim=1)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_per_sample + 1e-8)

        loss = -mean_log_prob_pos[pos_per_sample > 0].mean()
        return loss


class PrototypeBank(nn.Module):
    """
    类原型存储库

    为每个类别维护一个可学习的原型向量，使用动量更新。
    针对长尾类支持原型插值增强。
    """

    def __init__(
        self,
        n_classes: int,
        d_model: int = 256,
        momentum: float = 0.99,
        temperature: float = 0.1,
        tail_threshold: int = 10,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model
        self.momentum = momentum
        self.tau = temperature
        self.tail_threshold = tail_threshold

        # 原型向量
        self.prototypes = nn.Parameter(
            torch.randn(n_classes, d_model) * 0.02, requires_grad=False
        )
        # 每个类别的样本计数器（动量更新用）
        self.register_buffer('class_counts', torch.zeros(n_classes))
        # 类别之间的相似度（用于插值）
        self.register_buffer('proto_similarity', torch.eye(n_classes))

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor):
        """动量更新原型向量"""
        for c in range(self.n_classes):
            mask_c = (labels == c)
            if mask_c.sum() > 0:
                batch_centroid = features[mask_c].mean(dim=0)
                batch_centroid = F.normalize(batch_centroid, dim=0)
                self.prototypes[c] = (
                    self.momentum * self.prototypes[c]
                    + (1 - self.momentum) * batch_centroid
                )
                self.prototypes[c] = F.normalize(self.prototypes[c], dim=0)
                self.class_counts[c] += mask_c.sum()

    @torch.no_grad()
    def update_similarity(self):
        """更新原型间相似度矩阵"""
        proto_norm = F.normalize(self.prototypes, dim=1)
        self.proto_similarity = proto_norm @ proto_norm.T

    def get_augmented_prototypes(self) -> torch.Tensor:
        """
        对长尾类别进行原型插值增强。

        尾部类原型 = β * 自身原型 + (1-β) * 最近头部类原型
        β 随训练推移增加（tail_head_ratio 控制）
        """
        prototypes_aug = self.prototypes.clone()

        for c in range(self.n_classes):
            count = self.class_counts[c].item()
            if count > 0 and count <= self.tail_threshold:
                # 找到最近的头部类
                head_mask = self.class_counts > self.tail_threshold * 5
                if head_mask.sum() > 0:
                    head_sims = self.proto_similarity[c][head_mask]
                    nearest_head = torch.where(head_mask)[0][head_sims.argmax()]
                    # 插值: 权重偏向自身原型
                    beta = min(0.3 + 0.6 * (count / self.tail_threshold), 0.9)
                    prototypes_aug[c] = F.normalize(
                        beta * self.prototypes[c] + (1 - beta) * self.prototypes[nearest_head],
                        dim=0
                    )
        return prototypes_aug

    def forward(self, features: torch.Tensor, use_aug: bool = False) -> torch.Tensor:
        """
        原型分类 logits
        features: [B, d_model]
        返回: [B, n_classes]
        """
        proto = self.get_augmented_prototypes() if use_aug else self.prototypes
        features_norm = F.normalize(features, dim=1)
        proto_norm = F.normalize(proto, dim=1)
        return features_norm @ proto_norm.T / self.tau


class DCPN(nn.Module):
    """
    双重对比原型网络

    组合实例级监督对比 + 类原型对比
    """

    def __init__(
        self,
        n_classes: int = 14,
        d_model: int = 256,
        d_proj: int = 128,
        temperature_inst: float = 0.07,
        temperature_proto: float = 0.1,
        proto_momentum: float = 0.99,
        tail_threshold: int = 10,
    ):
        super().__init__()
        self.proj_head = ProjectionHead(d_model, d_proj, d_proj)
        self.supcon_loss = SupConLoss(temperature_inst)
        self.proto_bank = PrototypeBank(
            n_classes, d_proj, proto_momentum, temperature_proto, tail_threshold
        )
        self.n_classes = n_classes

    def forward(self, h_fused: torch.Tensor):
        """投影到对比空间"""
        return self.proj_head(h_fused)

    def compute_instance_loss(
        self, z: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """实例级监督对比损失"""
        return self.supcon_loss(z, labels)

    def compute_prototype_loss(
        self, z: torch.Tensor, labels: torch.Tensor, use_aug: bool = False
    ) -> torch.Tensor:
        """原型级对比损失"""
        logits = self.proto_bank(z, use_aug)
        return F.cross_entropy(logits, labels)

    def update_prototypes(self, z: torch.Tensor, labels: torch.Tensor):
        """动量更新原型"""
        self.proto_bank.update(z, labels)
        self.proto_bank.update_similarity()

    def get_prototype_logits(
        self, h_fused: torch.Tensor, use_aug: bool = False
    ) -> torch.Tensor:
        """获取原型分类 logits（用于下游分类器校准）"""
        z = self.proj_head(h_fused)
        return self.proto_bank(z, use_aug)
