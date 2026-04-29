"""
TrojViT/utils.py
----------------
工具函数集合，供 main_patch_vit.py、finetune_tiny_imagenet.py
以及上层防御测试文件 evaluation_test.py 统一 import。

新增部分（相比原版）
--------------------
- Tiny-ImageNet 数据集加载（get_loaders_* 系列）
- TrojViTCleanDataset / TrojViTBackdoorDataset
- load_trojvit_model / load_trojvit_trigger / build_trojvit_datasets
- trojvit_evaluate_cda_asr
- trojvit_detection_eval / trojvit_sample_detection_eval
"""

import json
import os
import shutil
import sys
import types
from os import path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from collections import Counter

# --------------------------------------------------------------------------
# 归一化常数（ImageNet，Tiny-ImageNet 同样适用）
# --------------------------------------------------------------------------
mu  = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

PATCH_SIZE   = 16
NUM_CLASSES  = 200          # Tiny-ImageNet


# ==========================================================================
# 通用工具
# ==========================================================================

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def PCGrad(atten_grad, ce_grad, sim, shape):
    """Gradient surgery：从 atten_grad 中移除与 ce_grad 冲突的分量。"""
    pcgrad       = atten_grad[sim < 0]
    temp_ce_grad = ce_grad[sim < 0]
    dot_prod     = torch.mul(pcgrad, temp_ce_grad).sum(dim=-1)
    dot_prod     = dot_prod / torch.norm(temp_ce_grad, dim=-1)
    pcgrad       = pcgrad - dot_prod.view(-1, 1) * temp_ce_grad
    atten_grad[sim < 0] = pcgrad
    return atten_grad.view(shape)


def shift_image(image, range, mu_val, std_val, patch_size=16):
    batch_size, channel, h, w = image.shape
    h_range, w_range = range
    new_h    = h + 2 * h_range * patch_size
    new_w    = w + 2 * w_range * patch_size
    new_image = torch.zeros([batch_size, channel, new_h, new_w]).cuda()
    new_image = (new_image - mu_val) / std_val
    shift_h   = np.random.randint(-h_range, h_range + 1)
    shift_w   = np.random.randint(-w_range, w_range + 1)
    new_image[:, :,
              h_range * patch_size: h + h_range * patch_size,
              w_range * patch_size: w + w_range * patch_size] = image.detach()
    h_start   = (h_range + shift_h) * patch_size
    w_start   = (w_range + shift_w) * patch_size
    return new_image[:, :, h_start: h_start + h, w_start: w_start + w]


# ==========================================================================
# Tiny-ImageNet 验证集整理
# ==========================================================================

def organise_tiny_imagenet_val(data_root: str) -> str:
    """
    将 Tiny-ImageNet val 集整理为 ImageFolder 兼容格式。
    原始结构：val/images/<flat> + val_annotations.txt
    整理后：  val_organised/<class_id>/<img>
    仅首次运行时执行，后续直接返回已整理路径。
    """
    organised = os.path.join(data_root, 'val_organised')
    if os.path.isdir(organised):
        return organised

    print('[utils] Organising Tiny-ImageNet val split (one-time) ...')
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

    print(f'[utils] Val split organised -> {organised}')
    return organised


def _build_tiny_imagenet_val(data_dir, img_size, crop_size, mu_val, std_val):
    val_dir   = organise_tiny_imagenet_val(data_dir)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mu_val, std=std_val),
    ])
    return datasets.ImageFolder(val_dir, transform=transform)


def _build_tiny_imagenet_train(data_dir, img_size, crop_size, mu_val, std_val):
    train_dir = os.path.join(data_dir, 'train')
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mu_val, std=std_val),
    ])
    return datasets.ImageFolder(train_dir, transform=transform)


# ==========================================================================
# 数据加载器（供 main_patch_vit.py 使用）
# ==========================================================================

def get_loaders(args):
    """触发器生成用：val 集前 384 条。"""
    args.mu  = mu
    args.std = std
    if getattr(args, 'dataset', 'ImageNet') == 'Tiny-ImageNet':
        val_dataset = _build_tiny_imagenet_val(
            args.data_dir, args.img_size, args.crop_size, mu, std)
    elif getattr(args, 'dataset', 'ImageNet') == 'Caltech101':
        val_dataset = _build_caltech101(
            args.data_dir, args.img_size, args.crop_size, mu, std, split='val')
    else:
        valdir      = path.join(args.data_dir, 'val')
        val_dataset = datasets.ImageFolder(
            valdir, transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.CenterCrop(args.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mu, std=std),
            ]))

    subset_size = min(384, len(val_dataset))
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(val_dataset, np.arange(subset_size)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


def get_loaders_test(args):
    """完整 val 集（用于精度评估）。"""
    args.mu  = mu
    args.std = std
    if getattr(args, 'dataset', 'ImageNet') == 'Tiny-ImageNet':
        val_dataset = _build_tiny_imagenet_val(
            args.data_dir, args.img_size, args.crop_size, mu, std)
    elif getattr(args, 'dataset', 'ImageNet') == 'Caltech101':
        val_dataset = _build_caltech101(
            args.data_dir, args.img_size, args.crop_size, mu, std, split='val')
    else:
        valdir      = path.join(args.data_dir, 'val')
        val_dataset = datasets.ImageFolder(
            valdir, transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.CenterCrop(args.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mu, std=std),
            ]))
    return torch.utils.data.DataLoader(
        val_dataset, batch_size=16, shuffle=True,
        num_workers=args.workers, pin_memory=True)


def get_loaders_testonebatch(args):
    args.mu  = mu
    args.std = std
    if getattr(args, 'dataset', 'ImageNet') == 'Tiny-ImageNet':
        val_dataset = _build_tiny_imagenet_val(
            args.data_dir, args.img_size, args.crop_size, mu, std)
    elif getattr(args, 'dataset', 'ImageNet') == 'Caltech101':
        val_dataset = _build_caltech101(
            args.data_dir, args.img_size, args.crop_size, mu, std, split='val')
    else:
        valdir      = path.join(args.data_dir, 'val')
        val_dataset = datasets.ImageFolder(
            valdir, transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.CenterCrop(args.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mu, std=std),
            ]))
    return torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


def get_loaders_test_small(args):
    """小 val 子集（1024 条，Trojan 插入阶段评估用）。"""
    args.mu  = mu
    args.std = std
    if getattr(args, 'dataset', 'ImageNet') == 'Tiny-ImageNet':
        val_dataset = _build_tiny_imagenet_val(
            args.data_dir, args.img_size, args.crop_size, mu, std)
    elif getattr(args, 'dataset', 'ImageNet') == 'Caltech101':
        val_dataset = _build_caltech101(
            args.data_dir, args.img_size, args.crop_size, mu, std, split='val')
    else:
        valdir      = path.join(args.data_dir, 'val')
        val_dataset = datasets.ImageFolder(
            valdir, transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.CenterCrop(args.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mu, std=std),
            ]))
    subset_size = min(1024, len(val_dataset))
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(val_dataset, np.arange(subset_size)),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)


# ==========================================================================
# TrojViT Dataset 类
# ==========================================================================

class TrojViTCleanDataset(Dataset):
    """Tiny-ImageNet 验证集干净版本。"""

    def __init__(self, val_dir: str, transform=None):
        self.base = ImageFolder(val_dir, transform=transform)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx]


class TrojViTBackdoorDataset(Dataset):
    """
    将 TrojViT 生成的触发器嵌入到 Tiny-ImageNet 验证集图像中，
    构造后门样本数据集。每个样本返回 (triggered_image, target_label)。

    Parameters
    ----------
    val_dir      : ImageFolder 格式的验证集目录
    delta        : Tensor [B, C, H, W]  触发器扰动（trigger_delta.pt）
    mask         : Tensor [B, 1, H, W]  patch 掩码（trigger_mask.pt）
    patch_index  : ndarray [B, num_patch]  patch 索引（trigger_patch_index.npy）
    target_class : 后门目标类别
    patch_size   : 每个 patch 的边长（像素）
    transform    : 图像预处理
    """

    def __init__(self, val_dir, delta, mask, patch_index,
                 target_class, patch_size=16, transform=None):
        self.base         = ImageFolder(val_dir, transform=transform)
        self.delta        = delta
        self.mask         = mask
        self.patch_index  = patch_index
        self.target_class = target_class
        self.patch_size   = patch_size
        self.trig_batch   = delta.shape[0]   # trigger batch 大小，超出时取余

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        img    = img.clone()

        trig_idx           = idx % self.trig_batch
        patch_num_per_line = img.shape[-1] // self.patch_size

        for pid in self.patch_index[trig_idx]:
            row = (pid // patch_num_per_line) * self.patch_size
            col = (pid  % patch_num_per_line) * self.patch_size
            img[:, row:row + self.patch_size, col:col + self.patch_size] = \
                self.delta[trig_idx, :,
                           row:row + self.patch_size,
                           col:col + self.patch_size]

        return img, self.target_class


# ==========================================================================
# TrojViT 模型加载
# ==========================================================================

def _deit_forward_single(self, x):
    """DeiT forward 补丁：始终返回 (cls+dist)/2 单个 tensor。"""
    x, x_dist = self.forward_features(x)
    return (self.head(x) + self.head_dist(x_dist)) / 2


def load_trojvit_model(network: str,
                       ckpt_path: str,
                       num_classes: int = NUM_CLASSES,
                       device: str = 'cuda') -> nn.Module:
    """
    加载 TrojViT 攻击产生的木马模型。

    Parameters
    ----------
    network     : 'DeiT-T' | 'DeiT-S' | 'DeiT-B' | 'ViT'
    ckpt_path   : .pth 文件路径（含 'state_dict' key）
    num_classes : 分类头输出维度
    device      : 运行设备
    """
    # 延迟 import，避免在非 TrojViT 环境中引入不必要依赖
    _trojvit_dir = os.path.dirname(__file__)
    if _trojvit_dir not in sys.path:
        sys.path.insert(0, _trojvit_dir)

    from models.DeiT import (deit_tiny_patch16_224,
                              deit_small_patch16_224,
                              deit_base_patch16_224)
    from timm import create_model as _timm_create

    if network == 'DeiT-T':
        model = deit_tiny_patch16_224(pretrained=False)
    elif network == 'DeiT-S':
        model = deit_small_patch16_224(pretrained=False)
    elif network == 'DeiT-B':
        model = deit_base_patch16_224(pretrained=False)
    elif network == 'ViT':
        model = _timm_create('vit_base_patch16_224', pretrained=False)
    else:
        raise ValueError(f'[utils] Unknown TrojViT network: {network}')

    # 替换分类头
    in_f       = model.head.in_features
    model.head = nn.Linear(in_f, num_classes)
    if hasattr(model, 'head_dist') and model.head_dist is not None:
        model.head_dist = nn.Linear(in_f, num_classes)
        model.forward   = types.MethodType(_deit_forward_single, model)

    # 加载权重
    ckpt  = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'[utils] Missing keys ({len(missing)}): {missing[:3]}')
    if unexpected:
        print(f'[utils] Unexpected keys ({len(unexpected)}): {unexpected[:3]}')

    model = model.to(device).eval()
    print(f'[utils] TrojViT model loaded  network={network}  ckpt={ckpt_path}')
    return model


# ==========================================================================
# TrojViT 触发器加载 & 数据集构建
# ==========================================================================

def load_trojvit_trigger(attack_dir: str):
    """
    从 TrojViT 攻击输出目录读取所有触发器文件。

    Returns
    -------
    delta       : Tensor  [B, C, H, W]
    mask        : Tensor  [B, 1, H, W]
    patch_index : ndarray [B, num_patch]
    config      : dict    trigger_config.json 的内容
    """
    delta       = torch.load(os.path.join(attack_dir, 'trigger_delta.pt'),
                              map_location='cpu')
    mask        = torch.load(os.path.join(attack_dir, 'trigger_mask.pt'),
                              map_location='cpu')
    patch_index = np.load(os.path.join(attack_dir, 'trigger_patch_index.npy'))

    with open(os.path.join(attack_dir, 'trigger_config.json'), 'r') as f:
        config = json.load(f)

    print(f'[utils] Trigger loaded from {attack_dir}')
    print(f'        delta={tuple(delta.shape)}  '
          f'patch_index={patch_index.shape}  '
          f'target_class={config.get("target_class")}')
    return delta, mask, patch_index, config

def _build_caltech101(data_root, img_size, crop_size, mu_val, std_val,
                      split='train', val_ratio=0.2, seed=42):
    """
    Caltech101 无官方划分，按 val_ratio 切分 train/val。
    split='train' 返回训练子集，split='val' 返回验证子集。
    """
    from torchvision.datasets import Caltech101
    from torch.utils.data import random_split

    categories_dir = os.path.join(data_root, 'caltech-101', '101_ObjectCategories')
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mu_val, std=std_val),
    ])
    full_ds  = datasets.ImageFolder(categories_dir, transform=transform)
    n_total  = len(full_ds)
    n_val    = int(n_total * val_ratio)
    n_train  = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(n_total), [n_train, n_val],
                                      generator=generator)
    if split == 'train':
        return torch.utils.data.Subset(full_ds, list(train_idx))
    else:
        return torch.utils.data.Subset(full_ds, list(val_idx))

def build_trojvit_datasets(attack_dir: str,
                            data_root: str,
                            subset_size: int = None):
    """
    构建 TrojViT 干净测试集和后门测试集。

    Parameters
    ----------
    attack_dir  : TrojViT 攻击输出目录（含 trigger_*.pt/npy/json）
    data_root   : Tiny-ImageNet 根目录
    subset_size : 各数据集取前 N 条；None 表示使用全集

    Returns
    -------
    test_data_clean    : Dataset
    test_data_backdoor : Dataset
    config             : dict
    """
    delta, mask, patch_index, config = load_trojvit_trigger(attack_dir)
    target_class = config['target_class']
    img_size     = config.get('img_size',  224)
    crop_size    = config.get('crop_size', 224)
    mean         = config.get('mu',  mu)
    std_val      = config.get('std', std)

    # val_dir   = organise_tiny_imagenet_val(data_root)

    dataset_name = config.get('dataset', 'Tiny-ImageNet')
    if dataset_name == 'Tiny-ImageNet':
        val_dir = organise_tiny_imagenet_val(data_root)
    elif dataset_name == 'Caltech101':
        val_dir = os.path.join(data_root, 'caltech-101', '101_ObjectCategories')
    else:
        raise ValueError(f'Unknown dataset in trigger_config: {dataset_name}')

    transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std_val),
    ])

    clean_ds    = TrojViTCleanDataset(val_dir, transform=transform)
    backdoor_ds = TrojViTBackdoorDataset(
        val_dir, delta, mask, patch_index,
        target_class=target_class,
        patch_size=config.get('patch_size', PATCH_SIZE),
        transform=transform)

    if subset_size is not None:
        clean_size = int(len(clean_ds) * subset_size)
        backdoor_size = len(clean_ds) - clean_size  # 剩余的就是1/5

        # 使用 Subset 按顺序拆分数据集
        clean_indices = list(range(clean_size))  # 前 4/5 的索引
        # clean_indices = list(range(1000))
        backdoor_indices = list(range(clean_size, len(clean_ds)))  # 后 1/5 的索引

        clean_ds    = Subset(clean_ds,    clean_indices)
        backdoor_ds = Subset(backdoor_ds, backdoor_indices)

    print(f'[utils] clean={len(clean_ds)}  backdoor={len(backdoor_ds)}')
    return clean_ds, backdoor_ds, config


# ==========================================================================
# TrojViT 评估函数（供 evaluation_test.py 调用）
# ==========================================================================

@torch.no_grad()
def trojvit_evaluate_cda_asr(model, clean_dataset, backdoor_dataset,
                              batch_size=64, device='cuda'):
    """
    计算木马模型的 CDA（干净数据准确率）和 ASR（攻击成功率）。

    Returns: (cda: float, asr: float)
    """
    model.eval()

    def _run(dataset):
        loader  = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
        correct = total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out    = model(imgs)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            preds  = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        return correct / total if total > 0 else 0.

    cda = _run(clean_dataset)
    asr = _run(backdoor_dataset)   # backdoor dataset 的 label 已经是 target_class
    print(f'[utils] CDA={cda*100:.2f}%  ASR={asr*100:.2f}%')
    return cda, asr


def trojvit_detection_eval(detector, test_data_backdoor, test_data_clean,
                            batch_size=1):
    """
    用 BackdoorDetector 对 TrojViT 样本进行检测，输出混淆矩阵 + TPR/FPR。

    Returns: dict with keys TP, FN, FP, TN, TPR, FPR,
                            results_backdoor, results_clean
    """
    print('\n[utils] BackdoorDetector on backdoor samples ...')
    results_bd = detector.detect_dataset(test_data_backdoor, batch_size=batch_size)
    print('[utils] BackdoorDetector on clean samples ...')
    results_cl = detector.detect_dataset(test_data_clean,    batch_size=batch_size)

    c_bd = Counter(results_bd.values())
    c_cl = Counter(results_cl.values())
    TP, FN = c_bd[True],  c_bd[False]
    FP, TN = c_cl[True],  c_cl[False]
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.

    print(f'[utils] TP={TP} FN={FN} FP={FP} TN={TN}  '
          f'TPR={TPR:.4f}  FPR={FPR:.4f}')
    return dict(TP=TP, FN=FN, FP=FP, TN=TN, TPR=TPR, FPR=FPR,
                results_backdoor=results_bd, results_clean=results_cl)


def trojvit_sample_detection_eval(detection, test_data_backdoor, test_data_clean):
    """
    用 BackdoorSampleDetection 对 TrojViT 样本进行检测。

    Returns: dict with keys TP, FN, FP, TN, TPR, FPR
    """
    print('\n[utils] BackdoorSampleDetection on backdoor data ...')
    results_bd = detection.detect_backdoor_samples_dataset(dataset=test_data_backdoor)
    c_bd = Counter(results_bd.values())
    TP, FN = c_bd[True], c_bd[False]
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.
    print(f'[utils] TPR={TPR:.4f}  (TP={TP}, FN={FN})')

    print('[utils] BackdoorSampleDetection on clean data ...')
    results_cl = detection.detect_backdoor_samples_dataset(dataset=test_data_clean)
    c_cl = Counter(results_cl.values())
    FP, TN = c_cl[True], c_cl[False]
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.
    print(f'[utils] FPR={FPR:.4f}  (FP={FP}, TN={TN})')

    return dict(TP=TP, FN=FN, FP=FP, TN=TN, TPR=TPR, FPR=FPR)


# ==========================================================================
# 日志 & 计量工具（供 main_patch_vit.py 使用，与原版保持一致）
# ==========================================================================

def visualize_loss(loss_list, args):
    plt.figure()
    plt.plot(loss_list)
    os.makedirs('output/loss', exist_ok=True)
    plt.savefig('output/loss/loss{:.4f}_{}_{}_{}_{}_{}.png'.format(
        loss_list[-1], args.network, args.attack_learning_rate,
        args.train_attack_iters, args.step_size, args.gamma))
    plt.close()


class my_logger:
    def __init__(self, args):
        name = "{}_{}_{}_{}_{}.log".format(
            args.name, args.network, args.dataset,
            args.train_attack_iters, args.attack_learning_rate)
        args.name  = name
        self.name  = path.join(args.log_dir, name)
        os.makedirs(args.log_dir, exist_ok=True)
        with open(self.name, 'w') as f:
            print('\n'.join(['%s:%s' % item
                             for item in args.__dict__.items()
                             if item[0][0] != '_']), file=f)
            print('\n', file=f)

    def info(self, content):
        with open(self.name, 'a') as f:
            print(content)
            print(content, file=f)


class my_meter:
    def __init__(self):
        self.meter_list = {}

    def add_loss_acc(self, model_name, loss_dic: dict, correct_num, batch_size):
        if model_name not in self.meter_list:
            self.meter_list[model_name] = self.model_meter()
        self.meter_list[model_name].add_loss_acc(loss_dic, correct_num, batch_size)

    def clean_meter(self):
        for v in self.meter_list.values():
            v.clean_meter()

    def get_loss_acc_msg(self):
        msg = []
        for key, sub in self.meter_list.items():
            sub_loss = sub.get_loss()
            loss_msg = ' '.join(
                '{}: {:.4f}({:.4f})'.format(x, sub.last_loss[x], sub_loss[x])
                for x in sub_loss)
            msg.append('model:{} Loss:{} Acc:{:.4f}({:.4f})'.format(
                key, loss_msg, sub.last_acc, sub.get_acc()))
        return '\n'.join(msg)

    class model_meter:
        def __init__(self):
            self.loss_bag  = {}
            self.acc       = 0.
            self.count     = 0
            self.last_loss = {}
            self.last_acc  = 0.

        def add_loss_acc(self, loss_dic, correct_num, batch_size):
            for k, v in loss_dic.items():
                self.loss_bag[k] = self.loss_bag.get(k, 0.) + v * batch_size
            self.last_loss = loss_dic
            self.last_acc  = correct_num / batch_size
            self.acc      += correct_num
            self.count    += batch_size

        def get_loss(self):
            return {k: v / self.count for k, v in self.loss_bag.items()}

        def get_acc(self):
            return self.acc / self.count

        def clean_meter(self):
            self.__init__()
