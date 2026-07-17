# HECTO-E: Industrial Control Valve Intelligent Selection System

> **HECTO-E** = **H**ierarchical **E**ncoder with **C**ascaded **T**ransformer **O**utput and **E**nhanced Prototype Learning

面向工业控制阀智能选型的层次化编码器-级联Transformer输出与增强原型学习模型

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

---

## 项目简介 | Overview

本项目是**六盘山实验室**与**西安交通大学**合作的横向课题成果，面向工业控制阀选型任务。模型接收异质特征输入（数值+类别），输出 **21 个层次化选型参数**（产品系列 → 型号 → 详细规格）。

HECTO-E 模型针对工业控制阀选型中的三大挑战设计：
- **异质特征融合**：数值特征 + 类别特征，通过门控注意力融合实现统一表征
- **层次化输出**：3级自回归解码器（系列 → 型号 → 详细规格），带合法性掩码约束
- **长尾分布**：DCPN 双重对比原型网络处理小样本类别

## 架构概览 | Architecture

HECTO-E 由五大模块组成：

| 模块 | 文件 | 功能 |
|------|------|------|
| **MissingHandler** | `models/missing_handler.py` | 逐特征可学习缺失嵌入 + MFM 掩码预训练头 |
| **MGHFE** | `models/heterogeneous_encoder.py` | 数值流（1D-CNN三元组+MLP）+ 类别流（实体嵌入+TabTransformer）→ 跨模态门控融合 |
| **HAOD** | `models/hierarchical_decoder.py` | 3级自回归解码器：产品系列(14类) → 型号(64类，带合法性掩码) → 详细规格(17分类+3回归) |
| **DCPN** | `models/prototype_network.py` | 实例级监督对比学习 + 类别级原型动量更新 + 长尾原型插值增强 |
| **PhysicsConstraints** | `models/physics_constraints.py` | 材料兼容性 + 压力-温度包络 + 层次一致性惩罚 |

### 数据流

```
原始CSV (GBK) → DataPreprocessor
  → X_num [N,22], X_cat [N,24]
  → MGHFE(数值流 + 类别流) → 门控融合 → h_fused [B,256]
  → HAOD(h_fused) → 层次化输出
  → DCPN(h_fused) → 对比学习投影
  → 同方差不确定性加权多任务损失
```

## 项目结构 | Project Structure

```
hecto-e/
├── configs/
│   └── hecto_config.yaml          # 训练配置文件
├── data/
│   ├── preprocessing.py           # 数据预处理（GBK加载、单位统一、编码）
│   └── dataset.py                 # PyTorch Dataset（预训练/训练/评估三种模式）
├── models/
│   ├── hecto_model.py             # HECTO-E 完整模型组装
│   ├── heterogeneous_encoder.py   # MGHFE 异构编码器
│   ├── hierarchical_decoder.py    # HAOD 层次化解码器
│   ├── prototype_network.py       # DCPN 原型对比学习
│   ├── missing_handler.py         # 缺失值处理 + MFM 预训练
│   └── physics_constraints.py     # 物理约束层
├── utils/
│   └── metrics.py                 # 评估指标
├── initdata/
│   └── README.md                  # 数据格式说明（原始数据保密）
├── train.py                       # 三阶段训练主脚本
├── requirements.txt               # Python依赖
├── .gitignore
├── LICENSE
└── README.md
```

## 快速开始 | Quick Start

### 环境要求 | Prerequisites

- Python 3.8+
- CPU（无需GPU，8核即可）

### 安装 | Installation

```bash
git clone https://github.com/your-org/hecto-e.git
cd hecto-e
pip install -r requirements.txt
```

### 准备数据

原始数据为保密数据，请联系项目方获取。获取后将 CSV 文件放入 `initdata/` 目录，数据格式详见 `initdata/README.md`。

### 训练 | Training

```bash
# 三阶段全量训练
python train.py --config configs/hecto_config.yaml --stage all

# 单阶段训练
python train.py --config configs/hecto_config.yaml --stage pretrain   # MFM预训练
python train.py --config configs/hecto_config.yaml --stage joint      # 联合多任务
python train.py --config configs/hecto_config.yaml --stage finetune   # 原型微调

# 从检查点恢复
python train.py --resume checkpoints/joint_best.pt --stage joint
```

## 训练流程 | Training Stages

### 阶段一：MFM 预训练（100 epochs）
掩码特征建模（Masked Feature Modeling），随机掩码15%已知特征进行重构，学习缺失模式。

### 阶段二：联合训练（300 epochs）
所有任务联合优化，采用**同方差不确定性加权**自动平衡多任务损失，教师强制比例从1.0逐步衰减至0.7。

### 阶段三：原型微调（50 epochs）
冻结编码器，使用类别平衡采样微调DCPN原型网络，重点提升长尾类别性能。

## 数据集 | Dataset

原始数据为保密数据，未包含在仓库中。数据格式说明见 `initdata/README.md`。

输出标签层次：
- **产品系列**: 14类
- **产品型号**: 64类（受系列约束）
- **详细规格**: 17个分类输出 + 3个回归输出 |

## 实验结果 | Results

| 指标 | 数值 |
|------|------|
| 产品系列 Acc@1 | 98.58% |
| 产品型号 Acc@1 | 69.05% |

## 引用 | Citation

如果您使用了本项目，请引用：

```bibtex
@software{hecto-e-2025,
  title     = {HECTO-E: Hierarchical Encoder with Cascaded Transformer Output for Industrial Control Valve Selection},
  author    = {{六盘山实验室} \& {西安交通大学}},
  year      = {2025},
  url       = {https://github.com/your-org/hecto-e},
}
```

## 许可证 | License

[MIT License](./LICENSE)
