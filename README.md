# research-project
面向工业控制阀智能选型的深度学习模型
六盘山实验室 × 西安交通大学

## 整体架构
HECTO-E（Hierarchical Encoder with Cascaded Transformer Output and
Enhanced Prototype Learning）是一个面向工业控制阀智能选型的深度
学习模型，从历史选型数据中自动学习复杂特征交互与决策模式，实现
从工况参数到产品系列、型号、详细规格的端到端智能推荐。
<img width="988" height="642" alt="441904822792c26aba679be353f445ec" src="https://github.com/user-attachments/assets/47e202c0-ea25-44db-8175-245c7d9bf457" />


## 核心模块
• MGHFE：多粒度异质特征编码器，融合22维数值特征与24维类别特征

• MFM：缺失特征建模预训练，处理高达50%的特征缺失率

• HAOD：层次化自回归解码器，显式建模"系列→型号→规格"依赖

• DCPN：双重对比原型网络，缓解长尾分布（头部:尾部≈500:1）

• PICL：物理信息约束层，确保材质兼容性与层次一致性

## 性能
在76,344条真实选型数据上，产品系列Top-1准确率达98.6%。

## 运行
Python 3.11 + PyTorch 2.x (CPU) | 参数量 ~6.6M | 训练耗时 ~77分钟
