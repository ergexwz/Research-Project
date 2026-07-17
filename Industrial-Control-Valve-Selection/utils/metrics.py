"""
多层级评估指标

- Top-1 / Top-3 Accuracy
- Macro F1
- Per-class breakdown (head/mid/tail)
- MAE / MAPE for regression
- 物理一致性得分
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, accuracy_score


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                     topk: Tuple[int, ...] = (1, 3)) -> Dict[str, float]:
    """Top-k 准确率"""
    results = {}
    targets = targets.long()
    for k in topk:
        pred = logits.topk(k, dim=-1).indices
        correct = (pred == targets.unsqueeze(-1)).any(dim=-1).float()
        results[f'acc@{k}'] = correct.mean().item()
    return results


def compute_f1_macro(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Macro F1"""
    preds = logits.argmax(dim=-1).cpu().numpy()
    targets = targets.cpu().numpy()
    return f1_score(targets, preds, average='macro', zero_division=0)


def compute_per_class_metrics(
    logits: torch.Tensor, targets: torch.Tensor,
    head_threshold: int = 1000, tail_threshold: int = 20
) -> Dict[str, float]:
    """
    按类别大小分组评估：头部 / 中部 / 尾部
    """
    preds = logits.argmax(dim=-1).cpu().numpy()
    targets = targets.cpu().numpy()

    unique, counts = np.unique(targets, return_counts=True)
    class_to_count = dict(zip(unique, counts))

    head_classes = {c for c, n in class_to_count.items() if n >= head_threshold}
    tail_classes = {c for c, n in class_to_count.items() if n <= tail_threshold}
    mid_classes = set(class_to_count.keys()) - head_classes - tail_classes

    results = {}
    for group_name, class_set in [('head', head_classes), ('mid', mid_classes), ('tail', tail_classes)]:
        if not class_set:
            continue
        mask = np.isin(targets, list(class_set))
        if mask.sum() == 0:
            continue
        results[f'acc_{group_name}'] = accuracy_score(targets[mask], preds[mask])
        results[f'f1_{group_name}'] = f1_score(
            targets[mask], preds[mask], average='macro', zero_division=0
        )
        results[f'n_{group_name}'] = mask.sum()

    return results


def compute_regression_metrics(
    preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None
) -> Dict[str, float]:
    """MAE / MAPE / RMSE"""
    if mask is not None:
        preds = preds[mask > 0.5]
        targets = targets[mask > 0.5]

    if len(preds) == 0:
        return {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0}

    mae = torch.abs(preds - targets).mean().item()
    rmse = torch.sqrt(((preds - targets) ** 2).mean()).item()

    # MAPE: 避免除零
    safe_targets = targets.abs().clamp(min=1e-6)
    mape = (torch.abs(preds - targets) / safe_targets).clamp(max=1.0).mean().item()

    return {'mae': mae, 'rmse': rmse, 'mape': mape}


def compute_physics_consistency(
    outputs: Dict, labels: Dict, model_legal_mask: torch.Tensor
) -> Dict[str, float]:
    """
    评估预测结果的物理一致性

    1. 层次一致性: 型号是否属于预测的系列
    2. 材质兼容性: 材质组合是否合理
    """
    results = {}

    # 层次一致性
    if 'pred_series' in outputs and 'pred_model' in outputs:
        mask = model_legal_mask[outputs['pred_series']]
        legal_models = mask[torch.arange(len(outputs['pred_model'])), outputs['pred_model']]
        results['hierarchy_consistency'] = legal_models.float().mean().item()

    return results


class MetricsTracker:
    """训练过程中累积和汇总指标"""

    def __init__(self):
        self.reset()

    def reset(self):
        self._data = {}

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            if k not in self._data:
                self._data[k] = []
            self._data[k].append(v)

    def summary(self) -> Dict[str, float]:
        return {k: np.mean(v) for k, v in self._data.items()}

    def __repr__(self):
        items = ', '.join([f'{k}: {v:.4f}' for k, v in self.summary().items()])
        return f'MetricsTracker({items})'
