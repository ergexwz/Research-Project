"""
HECTO-E 三阶段训练脚本

阶段一: MFM 预训练 — 缺失特征建模
阶段二: 联合训练 — 端到端多任务学习
阶段三: 原型微调 — Class-Balanced Prototype Fine-tuning
"""

import os
import sys
import argparse
import yaml
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.preprocessing import (
    DataPreprocessor, load_raw_data,
    CATEGORICAL_INPUT_COLS_LOW, CATEGORICAL_INPUT_COLS_HIGH,
    OUTPUT_SPECS_CLASSIFICATION_COLS, OUTPUT_SPECS_REGRESSION_COLS,
)
from data.dataset import ValveSelectionDataset, collate_fn_standard
from models.hecto_model import HECTOE
from utils.metrics import MetricsTracker, compute_accuracy, compute_f1_macro, compute_per_class_metrics


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build_model(config: dict, preprocessor: DataPreprocessor) -> HECTOE:
    """从配置和预处理器构建模型"""
    m = config['model']

    # 类别特征词汇量
    all_cat = CATEGORICAL_INPUT_COLS_LOW + CATEGORICAL_INPUT_COLS_HIGH
    cat_vocab_sizes = []
    for col in all_cat:
        if col in preprocessor.cat_vocab_sizes:
            cat_vocab_sizes.append(preprocessor.cat_vocab_sizes[col])
        else:
            cat_vocab_sizes.append(100)  # 兜底

    # Level 3 分类词汇量
    specs_cls_vocab = {}
    for col in OUTPUT_SPECS_CLASSIFICATION_COLS:
        if col in preprocessor.out_vocab_sizes:
            safe_name = col.replace('.', '_').replace('/', '_')
            specs_cls_vocab[safe_name] = preprocessor.out_vocab_sizes[col]

    # 型号→系列映射
    model_to_series = {}
    if 'model_by_series' in preprocessor.out_encoders:
        series_enc = preprocessor.out_encoders['产品系列']
        model_enc = preprocessor.out_encoders['产品型号']
        for series_key, model_le in preprocessor.out_encoders['model_by_series'].items():
            try:
                series_id = int(series_enc.transform([series_key])[0])
            except (ValueError, KeyError):
                series_id = 0
            for model_name in model_le.classes_:
                try:
                    model_id = int(model_enc.transform([model_name])[0])
                except (ValueError, KeyError):
                    continue
                if model_name != '__UNKNOWN__':
                    model_to_series[model_id] = series_id

    model = HECTOE(
        n_num_features=len(config.get('_n_num_features', 22)),
        n_triplets=m.get('n_triplets', 6),
        n_cat_features=len(cat_vocab_sizes),
        cat_vocab_sizes=cat_vocab_sizes,
        d_model=m.get('d_model', 256),
        d_emb=m.get('d_emb', 64),
        n_heads=m.get('n_heads', 8),
        n_transformer_layers=m.get('n_transformer_layers', 2),
        dropout=m.get('dropout', 0.1),
        n_series=m.get('n_series', 14),
        n_model_global=m.get('n_model_global', 64),
        specs_cls_vocab_sizes=specs_cls_vocab,
        n_specs_reg=m.get('n_specs_reg', 3),
        model_to_series=model_to_series,
        use_prototype=m.get('use_prototype', True),
        d_proj=m.get('d_proj', 128),
        temperature_inst=m.get('temperature_inst', 0.07),
        temperature_proto=m.get('temperature_proto', 0.1),
        proto_momentum=m.get('proto_momentum', 0.99),
        tail_threshold=m.get('tail_threshold', 10),
        compat_matrix=preprocessor.material_compat_matrix,
    )
    return model


def evaluate(model: HECTOE, loader: DataLoader, device: torch.device) -> dict:
    """在验证/测试集上评估"""
    model.eval()
    tracker = MetricsTracker()

    with torch.no_grad():
        for batch in tqdm(loader, desc='eval', leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(batch, mode='eval', teacher_forcing_ratio=0.0)

            # Level 1 准确率
            metrics = {}
            metrics.update(compute_accuracy(outputs['logits_series'], batch['y_series']))
            metrics['f1_series'] = compute_f1_macro(outputs['logits_series'], batch['y_series'])

            # Level 2 准确率
            metrics.update({
                k.replace('acc', 'acc_model'): v
                for k, v in compute_accuracy(outputs['logits_model'], batch['y_model']).items()
            })
            metrics['f1_model'] = compute_f1_macro(outputs['logits_model'], batch['y_model'])

            # 按类别分布
            metrics.update({
                f'series_{k}': v
                for k, v in compute_per_class_metrics(
                    outputs['logits_series'], batch['y_series']
                ).items()
            })

            tracker.update(metrics)

    return tracker.summary()


def stage_pretrain(model, train_loader, val_loader, config, device, writer):
    """阶段一: MFM 预训练"""
    print("\n" + "=" * 60)
    print("阶段一: MFM 预训练 (Masked Feature Modeling)")
    print("=" * 60)

    cfg = config['training']['pretrain']
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    best_loss = float('inf')
    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Pretrain Epoch {epoch+1}/{cfg["epochs"]}')
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(batch, mode='pretrain')
            losses = model.compute_losses(outputs, batch, mode='pretrain')

            optimizer.zero_grad()
            losses['L_total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()

            total_loss += losses['L_total'].item()
            pbar.set_postfix({'loss': losses['L_total'].item()})

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('pretrain/loss', avg_loss, epoch)
        print(f'Epoch {epoch+1}: avg_loss={avg_loss:.6f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(),
                       os.path.join(config['logging']['checkpoint_dir'], 'pretrain_best.pt'))

    print(f'MFM 预训练完成, best_loss={best_loss:.6f}')


def stage_joint(model, train_loader, val_loader, config, device, writer):
    """阶段二: 联合训练"""
    print("\n" + "=" * 60)
    print("阶段二: 联合训练 (Joint Multi-Task Training)")
    print("=" * 60)

    cfg = config['training']['joint']
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    # 学习率调度器: warmup + cosine
    total_steps = cfg['epochs'] * len(train_loader)
    warmup_steps = cfg['warmup_epochs'] * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Teacher forcing 衰减
    tf_start = cfg.get('teacher_forcing_start', 1.0)
    tf_end = cfg.get('teacher_forcing_end', 0.7)

    best_val_acc = 0.0
    global_step = 0

    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0.0
        tf_ratio = tf_start + (tf_end - tf_start) * (epoch / cfg['epochs'])

        pbar = tqdm(train_loader, desc=f'Joint Epoch {epoch+1}/{cfg["epochs"]}')
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(batch, mode='train', teacher_forcing_ratio=tf_ratio)
            losses = model.compute_losses(
                outputs, batch, mode='train',
                contrastive_weight=cfg['contrastive_weight'],
                proto_weight=cfg['proto_weight'],
                physics_weight=cfg['physics_weight'],
            )

            optimizer.zero_grad()
            losses['L_total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()
            scheduler.step()

            total_loss += losses['L_total'].item()
            global_step += 1

            if global_step % config['logging']['log_interval'] == 0:
                writer.add_scalar('joint/loss_total', losses['L_total'].item(), global_step)
                writer.add_scalar('joint/loss_series', losses['L_series'].item(), global_step)
                writer.add_scalar('joint/loss_model', losses['L_model'].item(), global_step)
                writer.add_scalar('joint/lr', scheduler.get_last_lr()[0], global_step)

            pbar.set_postfix({
                'loss': f'{losses["L_total"].item():.4f}',
                'tf': f'{tf_ratio:.2f}',
            })

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}: avg_loss={avg_loss:.4f}, tf_ratio={tf_ratio:.2f}')

        # 验证
        if (epoch + 1) % config['logging']['eval_interval'] == 0:
            val_metrics = evaluate(model, val_loader, device)
            writer.add_scalar('val/acc@1_series', val_metrics.get('acc@1', 0), epoch)
            writer.add_scalar('val/f1_series', val_metrics.get('f1_series', 0), epoch)
            print(f'  Val: acc@1={val_metrics.get("acc@1", 0):.4f}, f1={val_metrics.get("f1_series", 0):.4f}')

            if val_metrics.get('acc@1', 0) > best_val_acc:
                best_val_acc = val_metrics['acc@1']
                torch.save(model.state_dict(),
                           os.path.join(config['logging']['checkpoint_dir'], 'joint_best.pt'))

    print(f'联合训练完成, best_val_acc@1={best_val_acc:.4f}')


def stage_finetune(model, train_loader, val_loader, config, device, writer):
    """阶段三: 原型微调"""
    print("\n" + "=" * 60)
    print("阶段三: 原型微调 (Class-Balanced Prototype Fine-tuning)")
    print("=" * 60)

    cfg = config['training']['finetune']

    # 冻结编码器和解码器主干，仅训练原型和分类头
    for name, param in model.named_parameters():
        if 'dcpn' not in name and 'head_' not in name:
            if 'series_out_emb' not in name and 'model_out_emb' not in name:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)')

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )

    best_val_f1 = 0.0
    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Finetune Epoch {epoch+1}/{cfg["epochs"]}')
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(batch, mode='train', teacher_forcing_ratio=0.7)
            losses = model.compute_losses(
                outputs, batch, mode='train',
                contrastive_weight=0.5,
                proto_weight=0.5,
                physics_weight=0.05,
            )

            optimizer.zero_grad()
            losses['L_total'].backward()
            optimizer.step()

            total_loss += losses['L_total'].item()
            pbar.set_postfix({'loss': f'{losses["L_total"].item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}: avg_loss={avg_loss:.4f}')

        if (epoch + 1) % config['logging']['eval_interval'] == 0:
            val_metrics = evaluate(model, val_loader, device)
            writer.add_scalar('finetune/val_f1_series', val_metrics.get('f1_series', 0), epoch)
            print(f'  Val: f1={val_metrics.get("f1_series", 0):.4f}, '
                  f'tail_acc={val_metrics.get("series_acc_tail", 0):.4f}')

            if val_metrics.get('f1_series', 0) > best_val_f1:
                best_val_f1 = val_metrics['f1_series']
                torch.save(model.state_dict(),
                           os.path.join(config['logging']['checkpoint_dir'], 'finetune_best.pt'))

    print(f'原型微调完成, best_val_f1={best_val_f1:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/hecto_config.yaml')
    parser.add_argument('--stage', choices=['pretrain', 'joint', 'finetune', 'all'], default='all')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    set_seed(config['data']['random_seed'])
    device = get_device()
    print(f'Device: {device}')

    # 创建保存目录
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['logging']['tensorboard_dir'], exist_ok=True)

    # 加载与预处理数据
    print('加载数据...')
    df = load_raw_data(config['data']['raw_path'])

    preprocessor = DataPreprocessor()
    data = preprocessor.fit_transform(df)
    preprocessor.save(config['data']['preprocessor_path'])
    print(f'预处理器已保存至 {config["data"]["preprocessor_path"]}')

    config['_n_num_features'] = data['X_num'].shape[1]

    # 数据集划分
    n = data['X_num'].shape[0]
    train_end = int(n * config['data']['train_ratio'])
    val_end = train_end + int(n * config['data']['val_ratio'])

    indices = np.random.permutation(n)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    def split_data(data_dict, idx):
        return {k: v[idx] for k, v in data_dict.items() if isinstance(v, np.ndarray)}

    train_data = split_data(data, train_idx)
    val_data = split_data(data, val_idx)
    test_data = split_data(data, test_idx)

    # MFM 预训练: 需要特殊的 masking
    train_dataset_pretrain = ValveSelectionDataset(
        train_data, mode='pretrain',
        mfm_mask_ratio=config['training']['pretrain']['mfm_mask_ratio']
    )
    train_dataset = ValveSelectionDataset(train_data, mode='train')
    val_dataset = ValveSelectionDataset(val_data, mode='eval')
    test_dataset = ValveSelectionDataset(test_data, mode='eval')

    train_loader_pretrain = DataLoader(
        train_dataset_pretrain,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn_standard,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn_standard,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn_standard,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn_standard,
    )

    print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')

    # 构建模型
    print('构建 HECTO-E 模型...')
    model = build_model(config, preprocessor)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'总参数量: {total_params:,} / 可训练: {trainable_params:,}')

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f'从 {args.resume} 恢复模型权重')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(config['logging']['tensorboard_dir'], timestamp))

    # 三阶段训练
    if args.stage in ('pretrain', 'all'):
        stage_pretrain(model, train_loader_pretrain, val_loader, config, device, writer)

    if args.stage in ('joint', 'all'):
        stage_joint(model, train_loader, val_loader, config, device, writer)

    if args.stage in ('finetune', 'all'):
        stage_finetune(model, train_loader, val_loader, config, device, writer)

    # 最终测试评估
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)
    test_metrics = evaluate(model, test_loader, device)
    for k, v in test_metrics.items():
        print(f'  {k}: {v:.4f}')

    writer.close()
    print('训练完成!')


if __name__ == '__main__':
    main()
