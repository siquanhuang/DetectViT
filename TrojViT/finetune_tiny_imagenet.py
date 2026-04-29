"""
finetune_tiny_imagenet.py
--------------------------
Fine-tune a pretrained ImageNet-1K ViT / DeiT on Tiny-ImageNet (200 classes).

Usage examples
--------------
# DeiT-S  (推荐起点)
python finetune_tiny_imagenet.py --network DeiT-S

# DeiT-T  (最轻量)
python finetune_tiny_imagenet.py --network DeiT-T --epochs 100 --batch_size 64

# ViT-B/16
python finetune_tiny_imagenet.py --network ViT --lr 1e-4 --epochs 60

保存结构
--------
logs/finetune_<network>_TinyImageNet_<timestamp>/
    best_model.pth          <- 验证集 Top-1 最高的检查点（供 TrojViT 直接加载）
    last_model.pth          <- 最后一个 epoch 的权重
    finetune_log.csv        <- 每 epoch 的 train/val loss & acc
    finetune_config.json    <- 本次运行所有超参数
    training_curve.png      <- loss / acc 曲线图
    finetune_summary.json   <- 最终结果汇总
"""

import argparse
import csv
import json
import os
import shutil
import time
import types
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from timm import create_model

from models.DeiT import (deit_tiny_patch16_224,
                          deit_small_patch16_224,
                          deit_base_patch16_224)

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------
TINY_IMAGENET_ROOT = '/home/FedIPR-master/data/tiny-imagenet-200'
NUM_CLASSES        = 200
MEAN               = [0.485, 0.456, 0.406]
STD                = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# Tiny-ImageNet 验证集目录整理
# --------------------------------------------------------------------------
def organise_val_split(data_root: str) -> str:
    """
    将 Tiny-ImageNet val 集整理为 ImageFolder 兼容格式。
    原始结构：val/images/<flat> + val_annotations.txt
    整理后：  val_organised/<class_id>/<img>
    仅首次运行时执行，后续直接返回已整理路径。
    """
    organised = os.path.join(data_root, 'val_organised')
    if os.path.isdir(organised):
        return organised

    print('[finetune] Organising Tiny-ImageNet val split (one-time) ...')
    ann_file = os.path.join(data_root, 'val', 'val_annotations.txt')
    img_dir  = os.path.join(data_root, 'val', 'images')

    with open(ann_file, 'r') as f:
        for line in f:
            parts    = line.strip().split('\t')
            img_name = parts[0]
            class_id = parts[1]
            dst_dir  = os.path.join(organised, class_id)
            os.makedirs(dst_dir, exist_ok=True)
            src = os.path.join(img_dir, img_name)
            dst = os.path.join(dst_dir, img_name)
            if not os.path.exists(dst):
                shutil.copy(src, dst)

    print(f'[finetune] Val split organised -> {organised}')
    return organised


# --------------------------------------------------------------------------
# Data loaders
# --------------------------------------------------------------------------
def build_loaders(data_root, batch_size, workers, img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(data_root, 'train'), transform=train_transform)
    val_dataset   = datasets.ImageFolder(
        organise_val_split(data_root),    transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    return train_loader, val_loader, len(train_dataset), len(val_dataset)


# --------------------------------------------------------------------------
# Model builder
# 关键设计：给 DeiT 打补丁，使其在训练和推理时都返回单个 tensor，
# 彻底避免 (cls_logits, dist_logits) tuple 以及多卡 DataParallel 带来的问题。
# --------------------------------------------------------------------------
def _deit_forward_single(self, x):
    """替换 DeiT.forward，始终返回 (cls + dist) / 2 的单个 tensor。"""
    x, x_dist = self.forward_features(x)
    x      = self.head(x)
    x_dist = self.head_dist(x_dist)
    return (x + x_dist) / 2


def build_model(network: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    if network == 'DeiT-T':
        model = deit_tiny_patch16_224(pretrained=True)
    elif network == 'DeiT-S':
        model = deit_small_patch16_224(pretrained=True)
    elif network == 'DeiT-B':
        model = deit_base_patch16_224(pretrained=True)
    elif network == 'ViT':
        model = create_model('vit_base_patch16_224', pretrained=True)
    else:
        raise ValueError(f'Unsupported network: {network}')

    # ---- 替换分类头 ----
    in_features = model.head.in_features
    model.head  = nn.Linear(in_features, num_classes)
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)

    if hasattr(model, 'head_dist') and model.head_dist is not None:
        model.head_dist = nn.Linear(in_features, num_classes)
        nn.init.trunc_normal_(model.head_dist.weight, std=0.02)
        nn.init.zeros_(model.head_dist.bias)
        # 打补丁：统一输出为单 tensor，train/eval 模式下行为一致
        model.forward = types.MethodType(_deit_forward_single, model)
        print(f'[finetune] {network}: DeiT forward patched -> always returns single tensor')

    print(f'[finetune] {network}: head replaced (in={in_features} -> out={num_classes})')
    return model


# --------------------------------------------------------------------------
# Optimizer & scheduler
# --------------------------------------------------------------------------
def build_optimizer(model: nn.Module, args):
    head_params     = []
    backbone_params = []
    for name, param in model.named_parameters():
        if 'head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {'params': head_params,     'lr': args.lr,
         'name': 'head'},
        {'params': backbone_params, 'lr': args.lr * args.backbone_lr_scale,
         'name': 'backbone'},
    ]

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(param_groups, momentum=0.9,
                                    weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    return optimizer, scheduler


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch,
                    log_interval=50):
    model.train()
    total_loss = total_correct = total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        # forward 现在始终返回单个 tensor（DeiT 已打补丁，ViT 本来就是）
        out = model(images)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=1)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs             = labels.size(0)
        total_loss    += loss.item() * bs
        total_correct += (preds == labels).sum().item()
        total_samples += bs

        if (batch_idx + 1) % log_interval == 0:
            print(f'  Epoch {epoch:3d} [{batch_idx+1:4d}/{len(loader)}]  '
                  f'loss={total_loss/total_samples:.4f}  '
                  f'acc={total_correct/total_samples*100:.2f}%')

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct_top1 = total_correct_top5 = total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        # forward 始终返回单个 tensor
        out = model(images)
        logits = out[0] if isinstance(out, (tuple, list)) else out

        total_loss    += criterion(logits, labels).item() * labels.size(0)
        total_samples += labels.size(0)

        top1_pred = logits.argmax(dim=1)
        total_correct_top1 += (top1_pred == labels).sum().item()

        top5_pred = logits.topk(5, dim=1).indices
        total_correct_top5 += (top5_pred == labels.unsqueeze(1)).any(dim=1).sum().item()

    return (total_loss / total_samples,
            total_correct_top1 / total_samples,
            total_correct_top5 / total_samples)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_curves(history, save_path):
    epochs = list(range(1, len(history['train_loss']) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'],   label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, [a * 100 for a in history['train_acc']],  label='Train Top-1')
    axes[1].plot(epochs, [a * 100 for a in history['val_top1']],   label='Val Top-1')
    axes[1].plot(epochs, [a * 100 for a in history['val_top5']],   label='Val Top-5',
                 linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'[finetune] Training curve saved -> {save_path}')


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune ViT/DeiT on Tiny-ImageNet-200')

    parser.add_argument('--data_dir',
                        default=TINY_IMAGENET_ROOT, type=str)
    parser.add_argument('--img_size',          default=224,     type=int)
    parser.add_argument('--workers',           default=4,       type=int)
    parser.add_argument('--network',           default='DeiT-S', type=str,
                        choices=['DeiT-T', 'DeiT-S', 'DeiT-B', 'ViT'])
    parser.add_argument('--epochs',            default=100,     type=int)
    parser.add_argument('--batch_size',        default=64,      type=int)
    parser.add_argument('--optimizer',         default='adamw',
                        choices=['adamw', 'sgd'])
    parser.add_argument('--lr',                default=1e-3,    type=float)
    parser.add_argument('--backbone_lr_scale', default=0.1,     type=float)
    parser.add_argument('--weight_decay',      default=5e-4,    type=float)
    parser.add_argument('--warmup_epochs',     default=5,       type=int)
    parser.add_argument('--label_smoothing',   default=0.1,     type=float)
    parser.add_argument('--log_dir',           default='logs',  type=str)
    parser.add_argument('--save_freq',         default=10,      type=int,
                        help='每隔多少 epoch 保存一个中间检查点（0=不保存）')
    parser.add_argument('--log_interval',      default=50,      type=int)
    parser.add_argument('--seed',              default=42,      type=int)
    parser.add_argument('--device',            default='cuda',  type=str)
    parser.add_argument('--resume',            default='',      type=str,
                        help='从某个 .pth 检查点恢复训练')

    args = parser.parse_args()

    timestamp     = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.save_dir = os.path.join(
        args.log_dir,
        f'finetune_{args.network}_TinyImageNet_{timestamp}')
    os.makedirs(args.save_dir, exist_ok=True)
    return args


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args   = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print('=' * 65)
    print(f'  Fine-tune {args.network} on Tiny-ImageNet-200')
    print(f'  data_dir  : {args.data_dir}')
    print(f'  save_dir  : {args.save_dir}')
    print(f'  device    : {device}')
    print(f'  epochs    : {args.epochs}   batch_size: {args.batch_size}')
    print(f'  lr (head) : {args.lr}  '
          f'lr (backbone): {args.lr * args.backbone_lr_scale}')
    print('=' * 65)

    with open(os.path.join(args.save_dir, 'finetune_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ---- 数据 ----
    train_loader, val_loader, n_train, n_val = build_loaders(
        args.data_dir, args.batch_size, args.workers, args.img_size)
    print(f'[finetune] Train: {n_train}  Val: {n_val}')

    # ---- 模型（不使用 DataParallel，避免 tuple 输出问题）----
    model = build_model(args.network, NUM_CLASSES)
    model = model.to(device)

    print(f'[finetune] Params  total: '
          f'{sum(p.numel() for p in model.parameters()):,}  '
          f'trainable: '
          f'{sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    criterion                = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing).to(device)
    optimizer, scheduler     = build_optimizer(model, args)

    # ---- Warm-up scheduler ----
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: (ep + 1) / args.warmup_epochs
                              if ep < args.warmup_epochs else 1.0)

    # ---- 断点恢复 ----
    start_epoch = 0
    best_top1   = 0.0
    best_path   = os.path.join(args.save_dir, 'best_model.pth')
    history     = {'train_loss': [], 'train_acc': [],
                   'val_loss':   [], 'val_top1':  [], 'val_top5': []}

    if args.resume:
        print(f'[finetune] Resuming from {args.resume} ...')
        ckpt        = torch.load(args.resume, map_location='cpu')
        state       = ckpt.get('state_dict', ckpt)
        model.load_state_dict(state, strict=False)
        start_epoch = ckpt.get('epoch', 0)
        best_top1   = ckpt.get('best_top1', 0.0)
        print(f'[finetune] Resumed: epoch={start_epoch}  '
              f'best_top1={best_top1*100:.2f}%')

    # ---- CSV 日志 ----
    csv_path   = os.path.join(args.save_dir, 'finetune_log.csv')
    csv_file   = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'train_top1',
                         'val_loss', 'val_top1', 'val_top5',
                         'lr_head', 'lr_backbone', 'elapsed_sec'])

    # ---- 训练主循环 ----
    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        # -- 训练 --
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, args.log_interval)

        # -- 调度器（先 train，再 step，避免 warning）--
        if epoch <= args.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        lr_head     = optimizer.param_groups[0]['lr']
        lr_backbone = optimizer.param_groups[1]['lr']

        # -- 验证 --
        val_loss, val_top1, val_top5 = evaluate(
            model, val_loader, criterion, device)

        elapsed = time.time() - t0
        print(f'\n  Epoch {epoch:3d}/{args.epochs}  '
              f'train_loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%  '
              f'val_top1={val_top1*100:.2f}%  val_top5={val_top5*100:.2f}%  '
              f'lr_head={lr_head:.2e}  [{elapsed:.0f}s]\n')

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_top1'].append(val_top1)
        history['val_top5'].append(val_top5)

        csv_writer.writerow([
            epoch,
            f'{train_loss:.6f}', f'{train_acc*100:.4f}',
            f'{val_loss:.6f}',   f'{val_top1*100:.4f}',
            f'{val_top5*100:.4f}',
            f'{lr_head:.2e}',    f'{lr_backbone:.2e}',
            f'{elapsed:.1f}'])
        csv_file.flush()

        # -- 保存最佳模型 --
        if val_top1 > best_top1:
            best_top1 = val_top1
            torch.save({
                'state_dict':  model.state_dict(),
                'epoch':       epoch,
                'best_top1':   best_top1,
                'val_top5':    val_top5,
                'network':     args.network,
                'num_classes': NUM_CLASSES,
                'dataset':     'Tiny-ImageNet',
                'img_size':    args.img_size,
                'mean':        MEAN,
                'std':         STD,
            }, best_path)
            print(f'  * New best  val_top1={best_top1*100:.2f}%  -> {best_path}')

        # -- 中间检查点 --
        if args.save_freq > 0 and epoch % args.save_freq == 0:
            mid_path = os.path.join(
                args.save_dir, f'checkpoint_epoch{epoch:03d}.pth')
            torch.save({
                'state_dict':  model.state_dict(),
                'epoch':       epoch,
                'best_top1':   best_top1,
                'network':     args.network,
                'num_classes': NUM_CLASSES,
                'dataset':     'Tiny-ImageNet',
            }, mid_path)
            print(f'  [ckpt] -> {mid_path}')

    csv_file.close()

    # ---- 最后一个 epoch ----
    last_path = os.path.join(args.save_dir, 'last_model.pth')
    torch.save({
        'state_dict':  model.state_dict(),
        'epoch':       args.epochs,
        'best_top1':   best_top1,
        'network':     args.network,
        'num_classes': NUM_CLASSES,
        'dataset':     'Tiny-ImageNet',
        'img_size':    args.img_size,
        'mean':        MEAN,
        'std':         STD,
    }, last_path)
    print(f'[finetune] Last model saved -> {last_path}')

    plot_curves(history, os.path.join(args.save_dir, 'training_curve.png'))

    summary = {
        'network':        args.network,
        'dataset':        'Tiny-ImageNet',
        'num_classes':    NUM_CLASSES,
        'epochs':         args.epochs,
        'best_val_top1':  round(best_top1 * 100, 2),
        'final_val_top1': round(history['val_top1'][-1] * 100, 2),
        'final_val_top5': round(history['val_top5'][-1] * 100, 2),
        'best_model_path': best_path,
        'last_model_path': last_path,
    }
    with open(os.path.join(args.save_dir, 'finetune_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n' + '=' * 65)
    print(f'  Fine-tuning complete!')
    print(f'  Best  Val Top-1 : {best_top1*100:.2f}%')
    print(f'  Final Val Top-1 : {history["val_top1"][-1]*100:.2f}%')
    print(f'  All artefacts   : {args.save_dir}')
    print('=' * 65)
    print('\n[next step] TrojViT attack:')
    print(f'  python main_patch_vit.py \\')
    print(f'    --dataset Tiny-ImageNet \\')
    print(f'    --data_dir {args.data_dir} \\')
    print(f'    --network {args.network} \\')
    print(f'    --pretrained_ckpt {best_path} \\')
    print(f'    --target_class 2 --attack_mode Attention')


if __name__ == '__main__':
    main()
