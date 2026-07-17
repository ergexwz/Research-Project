"""
控制阀选型 PyTorch Dataset

支持：
  - MFM 预训练模式：随机遮蔽已知特征
  - 对比学习模式：同一 batch 返回两个经过不同 dropout 的视图
  - 多层级标签（Level 1/2/3）
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple


class ValveSelectionDataset(Dataset):
    """控制阀选型数据集"""

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        mode: str = 'train',       # 'pretrain' | 'train' | 'eval'
        mfm_mask_ratio: float = 0.15,
        contrastive: bool = False,
    ):
        self.X_num = torch.from_numpy(data['X_num'])
        self.mask_num = torch.from_numpy(data['mask_num'])
        self.X_cat = torch.from_numpy(data['X_cat'])
        self.mask_cat = torch.from_numpy(data['mask_cat'])

        self.Y_series = torch.from_numpy(data['series'])
        self.Y_model = torch.from_numpy(data['model'])
        self.Y_model_by_series = {
            k: torch.from_numpy(v) for k, v in data['model_by_series'].items()
        }
        self.Y_specs_cls = {
            k: torch.from_numpy(v) for k, v in data['specs_cls'].items()
        }
        self.Y_specs_reg = {
            k: torch.from_numpy(v) for k, v in data['specs_reg'].items()
        }

        self.mode = mode
        self.mfm_mask_ratio = mfm_mask_ratio
        self.contrastive = contrastive
        self.n_samples = len(self.X_num)
        self.n_num = self.X_num.shape[1]
        self.n_cat = self.X_cat.shape[1]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x_num = self.X_num[idx].clone()
        mask_num = self.mask_num[idx].clone()
        x_cat = self.X_cat[idx].clone()
        mask_cat = self.mask_cat[idx].clone()

        # MFM 预训练模式: 随机遮蔽已知特征用于重建
        if self.mode == 'pretrain':
            masked_targets_num = torch.zeros_like(x_num)
            masked_targets_cat = torch.zeros_like(x_cat)

            # 遮蔽数值特征
            known_num_indices = torch.where(mask_num > 0.5)[0]
            n_mask = max(1, int(len(known_num_indices) * self.mfm_mask_ratio))
            if len(known_num_indices) > 0:
                mask_idx = known_num_indices[torch.randperm(len(known_num_indices))[:n_mask]]
                masked_targets_num[mask_idx] = x_num[mask_idx]
                x_num[mask_idx] = 0.0
                mask_num[mask_idx] = 0.0  # 标记为"缺失"

            # 遮蔽类别特征
            known_cat_indices = torch.where(mask_cat > 0.5)[0]
            n_mask_cat = max(1, int(len(known_cat_indices) * self.mfm_mask_ratio))
            if len(known_cat_indices) > 0:
                mask_idx = known_cat_indices[torch.randperm(len(known_cat_indices))[:n_mask_cat]]
                masked_targets_cat[mask_idx] = x_cat[mask_idx].clone().to(masked_targets_cat.dtype)
                x_cat[mask_idx] = 0
                mask_cat[mask_idx] = 0.0

            return {
                'x_num': x_num,
                'mask_num': mask_num,
                'x_cat': x_cat,
                'mask_cat': mask_cat,
                'masked_targets_num': masked_targets_num,
                'masked_targets_cat': masked_targets_cat,
                'y_series': self.Y_series[idx],
                'y_model': self.Y_model[idx],
            }

        # 对比学习模式: 返回原始数据（batch 内两次 forward 产生两个视图）
        # 不需要在此处复制，model.forward 通过不同的 dropout_seed 实现
        item = {
            'x_num': x_num,
            'mask_num': mask_num,
            'x_cat': x_cat,
            'mask_cat': mask_cat,
            'y_series': self.Y_series[idx],
            'y_model': self.Y_model[idx],
        }
        # 添加 Level 3 标签
        for k, v in self.Y_specs_cls.items():
            item[f'y_specs_cls_{k}'] = v[idx]
        for k, v in self.Y_specs_reg.items():
            item[f'y_specs_reg_{k}'] = v[idx]

        # 部分 Level 3 为缺失 (__UNKNOWN__)，提供掩码
        item['valid_specs_mask'] = torch.ones(len(self.Y_specs_cls), dtype=torch.float32)

        return item


def collate_fn_standard(batch):
    """标准批次整理"""
    result = {}
    for key in batch[0].keys():
        tensors = [item[key] for item in batch]
        result[key] = torch.stack(tensors)
    return result
