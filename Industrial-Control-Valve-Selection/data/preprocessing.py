"""
控制阀选型数据预处理模块

功能：
  - GBK 编码读取与表头清洗
  - 列分组（输入数值 / 输入类别 / 输出分类 / 输出回归）
  - 异常值处理（inf, 负压, 负温等物理不可能值）
  - 单位统⼀归一化
  - 数值特征 log-压缩与标准化
  - 类别特征编码映射
  - 输出标签编码
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional
import pickle
import json


# ═══════════════════════════════════════════
# 列定义
# ═══════════════════════════════════════════

# 输入 — 数值列（22个）
NUMERICAL_INPUT_COLS = [
    # 三元组: CV
    '计算CV（最大）', '计算CV（正常）', '计算CV（最小）',
    # 三元组: 温度
    '入口温度（最大）', '入口温度（正常）', '入口温度（最小）',
    # 三元组: 流量
    '流量（最大）', '流量（正常）', '流量（最小）',
    # 三元组: 阀前压力
    '阀前压力（最大）', '阀前压力（正常）', '阀前压力（最小）',
    # 三元组: 阀后压力
    '阀后压力（最大）', '阀后压力（正常）', '阀后压力（最小）',
    # 三元组: 压差
    '压差（最大）', '压差（正常）', '压差（最小）',
    # 独立数值
    '关闭压力', '设计温度', '上游管道口径', '供气压力',
]

NUMERICAL_TRIPLET_GROUPS = [
    ('计算CV（最大）', '计算CV（正常）', '计算CV（最小）'),
    ('入口温度（最大）', '入口温度（正常）', '入口温度（最小）'),
    ('流量（最大）', '流量（正常）', '流量（最小）'),
    ('阀前压力（最大）', '阀前压力（正常）', '阀前压力（最小）'),
    ('阀后压力（最大）', '阀后压力（正常）', '阀后压力（最小）'),
    ('压差（最大）', '压差（正常）', '压差（最小）'),
]

# 输入 — 低基数类别列
CATEGORICAL_INPUT_COLS_LOW = [
    '流体状态', '阀门型式', '阀体型式', '流量特性', '泄露等级',
    '压力等级', '硬化保护', '法兰型式', '连接方式', '驱动方式',
    '环境温度', '防护等级', '阀体材质', '阀芯材质', '阀座材质',
    '阀杆材质', '填料', '温度单位', '流量单位', '压力单位', '制造品牌',
]

# 输入 — 高基数文本列
CATEGORICAL_INPUT_COLS_HIGH = [
    '位号', '流体名称', '管道材质',
]

# 输出 — 层次化
OUTPUT_SERIES_COL = '产品系列'          # Level 1
OUTPUT_MODEL_COL = '产品型号'           # Level 2
OUTPUT_SPECS_CLASSIFICATION_COLS = [    # Level 3 分类
    '公称通径', '压力等级.1', '法兰型式.1', '泄露等级.1',
    '流量特性.1', '阀盖型式', '阀体材质.1', '阀芯材质.1',
    '阀板材质', '阀座材质.1', '阀盖材质/轴材质', '密封环材质',
    '填料.1', '填料型式', '流向', '波纹管材质', '波纹管型式',
]
OUTPUT_SPECS_REGRESSION_COLS = [        # Level 3 回归
    '额定CV/KV', '阀座直径', '额定行程',
]

# 物理约束阈值
PHYSICS_CONSTRAINTS = {
    'pressure_min': -0.1,   # 压力不能为负（允许微小负值作为传感器噪声）
    'pressure_max': 20000,  # 最高 20000 MPa 为不合理噪点
    'temperature_min': -273.15,  # 绝对零度
    'temperature_max': 3000,     # 工业场景不可能超过
    'cv_min': 0,           # CV 不能为负
    'cv_max': 1e8,         # 异常上限
    'flow_min': 0,
    'flow_max': 1e10,
    'pipe_diameter_min': 1,   # 管径至少 1mm
    'pipe_diameter_max': 5000,
}


def load_raw_data(path: str) -> pd.DataFrame:
    """加载原始 CSV，清洗表头行"""
    df = pd.read_csv(path, encoding='gbk', header=1)
    # 过滤掉第一行是列名的脏数据（header row leaked as data）
    header_values = {'ID', '位号', '合同号', '管道材质', '制造品牌'}
    if 'ID' in df.columns:
        df = df[~df['ID'].isin(header_values)].copy()
    return df


def clean_numerical_values(df: pd.DataFrame) -> pd.DataFrame:
    """清洗数值特征：转 float，处理 inf，截断物理不可能值"""
    for col in NUMERICAL_INPUT_COLS + OUTPUT_SPECS_REGRESSION_COLS:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
        numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 物理约束
        if '压力' in col or '压差' in col:
            numeric = numeric.clip(
                PHYSICS_CONSTRAINTS['pressure_min'],
                PHYSICS_CONSTRAINTS['pressure_max']
            )
        elif '温度' in col or '设计温度' in col:
            numeric = numeric.clip(
                PHYSICS_CONSTRAINTS['temperature_min'],
                PHYSICS_CONSTRAINTS['temperature_max']
            )
        elif 'CV' in col:
            numeric = numeric.clip(
                PHYSICS_CONSTRAINTS['cv_min'],
                PHYSICS_CONSTRAINTS['cv_max']
            )
        elif '流量' in col:
            numeric = numeric.clip(
                PHYSICS_CONSTRAINTS['flow_min'],
                PHYSICS_CONSTRAINTS['flow_max']
            )
        elif '口径' in col or '通径' in col or '直径' in col:
            numeric = numeric.clip(
                PHYSICS_CONSTRAINTS['pipe_diameter_min'],
                PHYSICS_CONSTRAINTS['pipe_diameter_max']
            )
        df[col] = numeric
    return df


def unit_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """单位统一归一化"""
    # 压力单位统一 → MPa
    pressure_cols = [c for c in NUMERICAL_INPUT_COLS + ['关闭压力', '供气压力']
                     if '压力' in c or '压差' in c]

    # 从压力单位列推断每条记录的原始单位
    for col in pressure_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(float)
        # MPa.a 和 MPa.g 在数值上相同，区别在参考基准
        # kPa → MPa: /1000
        # bar → MPa: /10
        # psi → MPa: /145.038
        # kgf/cm² → MPa: /10.197
        # Pa → MPa: /1e6
        # mmHg → MPa: /7500.62
        mask_kpa = df['压力单位'].str.contains('kPa|KPa', na=False)
        mask_bar = df['压力单位'].str.contains('bar', na=False, case=False)
        mask_psi = df['压力单位'].str.contains('psi', na=False, case=False)
        mask_kgf = df['压力单位'].str.contains('kgf', na=False)
        mask_pa  = df['压力单位'].str.contains('Pa\\.g|Pa\\.a', na=False) & ~mask_kpa
        mask_mmhg = df['压力单位'].str.contains('mmHg', na=False, case=False)

        df.loc[mask_kpa, col] = df.loc[mask_kpa, col] / 1000.0
        df.loc[mask_bar, col] = df.loc[mask_bar, col] / 10.0
        df.loc[mask_psi, col] = df.loc[mask_psi, col] / 145.038
        df.loc[mask_kgf, col] = df.loc[mask_kgf, col] / 10.197
        df.loc[mask_pa, col]  = df.loc[mask_pa, col] / 1e6
        df.loc[mask_mmhg, col] = df.loc[mask_mmhg, col] / 7500.62

    # 温度单位统一 → °C
    # 数据中绝大部分已是 °C，少量其他单位保持原值标记即可
    temp_cols = ['入口温度（最大）', '入口温度（正常）', '入口温度（最小）', '设计温度']

    # 流量单位统一 → Nm³/h (保留为标记，不做硬转，因为介质不同难以统一)
    return df


def normalize_numerical_features(df: pd.DataFrame, scaler: Optional[StandardScaler] = None,
                                 fit: bool = True) -> Tuple[np.ndarray, np.ndarray, StandardScaler, Dict]:
    """
    数值特征处理: log-压缩 + StandardScaler
    CV 和流量使用 log1p 压缩量级跨度
    """
    stats = {}
    X_num = np.zeros((len(df), len(NUMERICAL_INPUT_COLS)), dtype=np.float32)
    mask_num = np.zeros((len(df), len(NUMERICAL_INPUT_COLS)), dtype=np.float32)

    for i, col in enumerate(NUMERICAL_INPUT_COLS):
        if col not in df.columns:
            continue
        vals = df[col].values.astype(np.float64)
        mask_num[:, i] = (~np.isnan(vals)).astype(np.float32)

        # log 压缩 CV 和流量
        if 'CV' in col or '流量' in col:
            vals = np.log1p(np.maximum(vals, 0))

        # 填充 NaN（F列）
        X_num[:, i] = np.nan_to_num(vals, nan=0.0).astype(np.float32)

        if fit:
            valid = vals[~np.isnan(vals)]
            stats[col] = {'mean': float(np.mean(valid)), 'std': float(np.std(valid) + 1e-8)}

    if fit:
        scaler = StandardScaler()
        scaler.fit(X_num)
    X_num_scaled = scaler.transform(X_num)

    return X_num_scaled.astype(np.float32), mask_num.astype(np.float32), scaler, stats


def encode_categorical_features(df: pd.DataFrame,
                                encoders: Optional[Dict[str, LabelEncoder]] = None,
                                fit: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, LabelEncoder], Dict[str, int]]:
    """
    类别特征编码
    返回: (X_cat_idxs, mask_cat, encoders, vocab_sizes)
    """
    all_cat_cols = CATEGORICAL_INPUT_COLS_LOW + CATEGORICAL_INPUT_COLS_HIGH

    if fit:
        encoders = {}
        vocab_sizes = {}
        for col in all_cat_cols:
            if col not in df.columns:
                continue
            encoder = LabelEncoder()
            vals = df[col].fillna('__MISSING__').astype(str).values
            encoder.fit(vals)
            encoders[col] = encoder
            vocab_sizes[col] = len(encoder.classes_)
    else:
        vocab_sizes = {col: len(enc.encoding_classes_) for col, enc in encoders.items()}

    X_cat = np.zeros((len(df), len(all_cat_cols)), dtype=np.int64)
    mask_cat = np.zeros((len(df), len(all_cat_cols)), dtype=np.float32)

    for i, col in enumerate(all_cat_cols):
        if col not in df.columns or col not in encoders:
            continue
        vals = df[col].fillna('__MISSING__').astype(str).values
        mask_cat[:, i] = (df[col].notna()).astype(np.float32).values
        X_cat[:, i] = encoders[col].transform(vals)

    return X_cat, mask_cat, encoders, vocab_sizes


def encode_output_labels(df: pd.DataFrame,
                         encoders: Optional[Dict[str, LabelEncoder]] = None,
                         fit: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, LabelEncoder], Dict[str, int]]:
    """
    编码输出标签
    返回: (labels_dict, encoders, vocab_sizes)
    labels_dict = {
        'series': [batch],       # Level 1
        'model': [batch],         # Level 2
        'model_by_series': {series_id: [batch]},  # Level 2 系列内索引
        'specs_cls': {col: [batch]},  # Level 3 分类
        'specs_reg': {col: [batch]},  # Level 3 回归
    }
    """
    all_output_cls = [OUTPUT_SERIES_COL, OUTPUT_MODEL_COL] + OUTPUT_SPECS_CLASSIFICATION_COLS
    all_output_reg = OUTPUT_SPECS_REGRESSION_COLS

    if fit:
        encoders = {}
        vocab_sizes = {}

        # 产品系列编码器
        series_vals = df[OUTPUT_SERIES_COL].fillna('__UNKNOWN__').astype(str).values
        le_series = LabelEncoder()
        le_series.fit(series_vals)
        encoders[OUTPUT_SERIES_COL] = le_series
        vocab_sizes[OUTPUT_SERIES_COL] = len(le_series.classes_)

        # 产品型号全局编码器
        model_vals = df[OUTPUT_MODEL_COL].fillna('__UNKNOWN__').astype(str).values
        le_model = LabelEncoder()
        le_model.fit(model_vals)
        encoders[OUTPUT_MODEL_COL] = le_model
        vocab_sizes[OUTPUT_MODEL_COL] = len(le_model.classes_)

        # 产品型号按系列编码器 (Level 2 conditioned on Level 1)
        encoders['model_by_series'] = {}
        vocab_sizes['model_by_series'] = {}
        for series_id, group in df.groupby(OUTPUT_SERIES_COL):
            series_key = str(series_id) if pd.notna(series_id) else '__UNKNOWN__'
            model_vals = group[OUTPUT_MODEL_COL].fillna('__UNKNOWN__').astype(str).values
            le = LabelEncoder()
            le.fit(model_vals)
            encoders['model_by_series'][series_key] = le
            vocab_sizes['model_by_series'][series_key] = len(le.classes_)

        # Level 3 分类编码器
        for col in OUTPUT_SPECS_CLASSIFICATION_COLS:
            if col not in df.columns:
                continue
            vals = df[col].fillna('__UNKNOWN__').astype(str).values
            le = LabelEncoder()
            le.fit(vals)
            encoders[col] = le
            vocab_sizes[col] = len(le.classes_)

    # 编码
    labels = {}
    labels['series'] = encoders[OUTPUT_SERIES_COL].transform(
        df[OUTPUT_SERIES_COL].fillna('__UNKNOWN__').astype(str).values
    ).astype(np.int64)

    labels['model'] = encoders[OUTPUT_MODEL_COL].transform(
        df[OUTPUT_MODEL_COL].fillna('__UNKNOWN__').astype(str).values
    ).astype(np.int64)

    # 系列内型号索引
    labels['model_by_series'] = {}
    series_vals = df[OUTPUT_SERIES_COL].fillna('__UNKNOWN__').astype(str).values
    model_vals = df[OUTPUT_MODEL_COL].fillna('__UNKNOWN__').astype(str).values
    for series_key, le in encoders['model_by_series'].items():
        idx = np.zeros(len(df), dtype=np.int64)
        mask = series_vals == series_key
        idx[mask] = le.transform(model_vals[mask])
        labels['model_by_series'][series_key] = idx

    # Level 3 分类标签
    labels['specs_cls'] = {}
    for col in OUTPUT_SPECS_CLASSIFICATION_COLS:
        if col not in df.columns or col not in encoders:
            continue
        vals = df[col].fillna('__UNKNOWN__').astype(str).values
        labels['specs_cls'][col] = encoders[col].transform(vals).astype(np.int64)

    # Level 3 回归标签
    labels['specs_reg'] = {}
    for col in OUTPUT_SPECS_REGRESSION_COLS:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors='coerce').values
        vals = np.nan_to_num(vals, nan=0.0).astype(np.float32)
        labels['specs_reg'][col] = vals

    return labels, encoders, vocab_sizes


def build_material_compatibility_matrix(labels: Dict[str, np.ndarray],
                                       encoders: Dict[str, LabelEncoder]) -> np.ndarray:
    """
    从训练数据统计材质共现矩阵
    返回: compat_matrix[body_mat, core_mat] = 条件概率
    """
    body_vocab = len(encoders['阀体材质.1'].classes_)
    core_vocab = len(encoders['阀芯材质.1'].classes_)

    cooccur = np.ones((body_vocab, core_vocab))  # Laplace 平滑
    body_labels = labels['specs_cls']['阀体材质.1']
    core_labels = labels['specs_cls']['阀芯材质.1']

    for b, c in zip(body_labels, core_labels):
        if b < body_vocab and c < core_vocab:
            cooccur[b, c] += 1

    # 行归一化 → 条件概率 P(core | body)
    compat = cooccur / (cooccur.sum(axis=1, keepdims=True) + 1e-8)
    return compat.astype(np.float32)


class DataPreprocessor:
    """完整预处理流水线"""

    def __init__(self):
        self.num_scaler: Optional[StandardScaler] = None
        self.num_stats: Dict = {}
        self.cat_encoders: Dict[str, LabelEncoder] = {}
        self.cat_vocab_sizes: Dict[str, int] = {}
        self.out_encoders: Dict[str, LabelEncoder] = {}
        self.out_vocab_sizes: Dict[str, int] = {}
        self.material_compat_matrix: Optional[np.ndarray] = None

    def fit_transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        df = clean_numerical_values(df)
        df = unit_normalize(df)

        X_num, mask_num, self.num_scaler, self.num_stats = normalize_numerical_features(df, fit=True)
        X_cat, mask_cat, self.cat_encoders, self.cat_vocab_sizes = encode_categorical_features(df, fit=True)

        labels, self.out_encoders, self.out_vocab_sizes = encode_output_labels(df, fit=True)
        self.material_compat_matrix = build_material_compatibility_matrix(labels, self.out_encoders)

        return {
            'X_num': X_num.astype(np.float32),
            'mask_num': mask_num.astype(np.float32),
            'X_cat': X_cat,
            'mask_cat': mask_cat,
            **labels,
        }

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        df = clean_numerical_values(df)
        df = unit_normalize(df)
        X_num, mask_num, _, _ = normalize_numerical_features(
            df, scaler=self.num_scaler, fit=False
        )
        X_cat, mask_cat, _, _ = encode_categorical_features(
            df, encoders=self.cat_encoders, fit=False
        )
        labels, _, _ = encode_output_labels(
            df, encoders=self.out_encoders, fit=False
        )
        return {
            'X_num': X_num.astype(np.float32),
            'mask_num': mask_num.astype(np.float32),
            'X_cat': X_cat,
            'mask_cat': mask_cat,
            **labels,
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'DataPreprocessor':
        with open(path, 'rb') as f:
            return pickle.load(f)
