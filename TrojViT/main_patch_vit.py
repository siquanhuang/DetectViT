"""
TrojViT – Patch-wise Trojan attack on Vision Transformers
Modified to support Tiny-ImageNet (200 classes) in addition to
ImageNet-1K, and to save all attack artefacts under
    logs/<dataset>_<network>_target<T>_<timestamp>/
so that downstream defence experiments can reload them directly.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import json
import numpy as np
import time
from timm import create_model
from adaptive_attack_DetectViT import adaptive_attack_loss, register_deit_hooks_for_reg

from models.DeiT import deit_base_patch16_224, deit_tiny_patch16_224, deit_small_patch16_224
from utils import (clamp, get_loaders, get_loaders_test, get_loaders_test_small,
                   my_logger, my_meter, PCGrad)

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test, attack_over_test_data

from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import random
from math import floor
import operator
import copy
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from datetime import datetime

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
patch_size = 16
high       = 100
targets    = 2          # target backdoor class index


# ---------------------------------------------------------------------------
# Number of classes per dataset  (used to set wb / wb1 / model head)
# ---------------------------------------------------------------------------
NUM_CLASSES = {
    'ImageNet':      1000,
    'Tiny-ImageNet': 200,
    'Caltech101':    102,   # torchvision Caltech101 含 BACKGROUND_Google 共102类
}

# embed_dim per network  (used to set wb / wb1)
EMBED_DIM = {
    'DeiT-T': 192,
    'DeiT-S': 384,
    'DeiT-B': 768,
    'ViT':    768,
}


# ---------------------------------------------------------------------------
# Trigger-generation attack (FGSM on selected patches)
# ---------------------------------------------------------------------------
class Attack(object):

    def __init__(self, dataloader, criterion=None, gpu_id=0,
                 epsilon=0.031, attack_method='pgd'):
        self.criterion     = nn.MSELoss()
        self.dataloader    = dataloader
        self.epsilon       = epsilon
        self.gpu_id        = gpu_id

        if attack_method == 'fgsm':
            self.attack_method = self.fgsm
        elif attack_method == 'pgd':
            self.attack_method = self.pgd
        elif attack_method == 'fgsm_patch':
            self.attack_method = self.fgsm_patch

    def update_params(self, epsilon=None, dataloader=None, attack_method=None):
        if epsilon is not None:
            self.epsilon = epsilon
        if dataloader is not None:
            self.dataloader = dataloader
        if attack_method is not None:
            if attack_method == 'fgsm':
                self.attack_method = self.fgsm
            elif attack_method == 'fgsm_patch':
                self.attack_method = self.fgsm_patch

    def fgsm_patch(self, model, data, max_patch_index, target, tar, ep,
                   data_min=0, data_max=1):
        model.eval()
        perturbed_data = data.clone()
        perturbed_data.requires_grad = True
        output = model.forward_features(perturbed_data)
        loss   = self.criterion(output[:, tar], target[:, tar])

        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()
        loss.backward(retain_graph=True)

        sign_data_grad = perturbed_data.grad.data.sign()
        perturbed_data.requires_grad = False

        with torch.no_grad():
            patch_num_per_line = int(data.size(-1) / patch_size)
            for j in range(data.size(0)):
                for index in max_patch_index[j]:
                    row    = (index // patch_num_per_line) * patch_size
                    column = (index  % patch_num_per_line) * patch_size
                    perturbed_data[j, :,
                                   row:row + patch_size,
                                   column:column + patch_size] -= (
                        ep * sign_data_grad[j, :,
                                            row:row + patch_size,
                                            column:column + patch_size])
        return perturbed_data


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def get_aug():
    parser = argparse.ArgumentParser(description='TrojViT – Patch-wise Trojan')

    # ---- identity / logging ----
    parser.add_argument('--name',    default='trojvit', type=str)
    parser.add_argument('--log_dir', default='logs',    type=str,
                        help='Root directory for all experiment logs')

    # ---- dataset ----
    parser.add_argument('--dataset', default='Tiny-ImageNet', type=str,
                        choices=['ImageNet', 'Tiny-ImageNet', 'Caltech101'])
    parser.add_argument('--data_dir', default='/data/tiny-imagenet-200/', type=str,
                        help='Path to dataset root (must contain train/ and val/)')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--crop_size',  default=224, type=int)
    parser.add_argument('--img_size',   default=224, type=int,
                        help='Resize target before CenterCrop '
                             '(Tiny-ImageNet native is 64×64, we upscale to 224)')
    parser.add_argument('--workers',    default=4,   type=int)
    parser.add_argument('--device',     default='cuda')

    # ---- model ----
    parser.add_argument('--network', default='DeiT-S', type=str,
                        choices=['DeiT-B', 'DeiT-S', 'DeiT-T', 'ViT',
                                 'ResNet152', 'ResNet50', 'ResNet18'])

    # ---- patch selection ----
    parser.add_argument('--dataset_size',     default=0.1, type=float)
    parser.add_argument('--patch_select',     default='Saliency', type=str,
                        choices=['Rand', 'Saliency', 'Attn', 'Fixed'])
    parser.add_argument('--num_patch',        default=9, type=int)
    parser.add_argument('--sparse_pixel_num', default=0, type=int)

    # ---- attack loss ----
    parser.add_argument('--attack_mode',       default='CE_loss',
                        choices=['CE_loss', 'Attention'], type=str)
    parser.add_argument('--atten_loss_weight', default=1,   type=float)
    parser.add_argument('--atten_select',      default=4,   type=int)
    parser.add_argument('--mild_l_2',          default=0.,  type=float)
    parser.add_argument('--mild_l_inf',        default=0.,  type=float)

    # ---- trigger optimisation ----
    parser.add_argument('--train_attack_iters',  default=250,  type=int)
    parser.add_argument('--random_sparse_pixel', action='store_true')
    parser.add_argument('--learnable_mask_stop', default=200,  type=int)
    parser.add_argument('--attack_learning_rate',default=0.22, type=float)
    parser.add_argument('--step_size',           default=10,   type=int)
    parser.add_argument('--gamma',               default=0.95, type=float)

    # ---- Trojan insertion ----
    parser.add_argument('--trojan_epochs',    default=200,   type=int)
    parser.add_argument('--trojan_lr',        default=0.01,  type=float)
    parser.add_argument('--distill_thresh',   default=0.003, type=float,
                        help='Parameter distillation threshold e')
    parser.add_argument('--trojan_loss_g',    default=0.5,   type=float,
                        help='Weight for clean-data loss during Trojan insertion')

    # ---- target class ----
    parser.add_argument('--target_class', default=2, type=int,
                        help='Backdoor target class index')

    # ---- pretrained checkpoint (e.g. from finetune_tiny_imagenet.py) ----
    parser.add_argument('--pretrained_ckpt', default='', type=str,
                        help='Path to a fine-tuned .pth checkpoint to load '
                             'before running the Trojan attack. '
                             'Expected key: "state_dict". '
                             'If empty, the standard ImageNet-1K pretrained '
                             'weights are used (head is re-initialised for '
                             'Tiny-ImageNet).')

    # ---- misc ----
    parser.add_argument('--seed', default=18, type=int)

    args = parser.parse_args()

    # ---- validation ----
    if args.mild_l_2 != 0 and args.mild_l_inf != 0:
        raise ValueError('Only one of mild_l_2 / mild_l_inf can be non-zero.')
    if args.mild_l_inf > 1:
        args.mild_l_inf /= 255.

    # ---- Tiny-ImageNet specific sanity check ----
    if args.dataset in NUM_CLASSES:
        n_cls = NUM_CLASSES[args.dataset]
        if args.target_class >= n_cls:
            raise ValueError(
                f'--target_class {args.target_class} >= '
                f'num_classes {n_cls} for {args.dataset}')

    # ---- build run-specific save directory ----
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name   = f"{args.dataset}_{args.network}_target{args.target_class}_{timestamp}"
    args.save_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(args.save_dir, exist_ok=True)

    return args


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_trigger_info(save_dir, delta, mask, max_patch_index, args):
    """
    Persist everything needed to reproduce / apply the trigger later.

    Saved files
    -----------
    trigger_delta.pt       – raw perturbation tensor  [B, C, H, W]
    trigger_mask.pt        – binary patch mask tensor  [B, 1, H, W]
    trigger_patch_index.npy– patch indices array       [B, num_patch]
    trigger_delta_vis.png  – first-sample visualisation (de-normalised)
    trigger_config.json    – hyper-params used to generate the trigger
    """
    os.makedirs(save_dir, exist_ok=True)

    torch.save(delta.detach().cpu(),       os.path.join(save_dir, 'trigger_delta.pt'))
    torch.save(mask.detach().cpu(),        os.path.join(save_dir, 'trigger_mask.pt'))
    np.save(os.path.join(save_dir, 'trigger_patch_index.npy'),
            max_patch_index.cpu().numpy() if isinstance(max_patch_index, torch.Tensor)
            else max_patch_index)

    # --- visual: de-normalise first sample and clamp to [0,1] ---
    mu_t  = torch.tensor(args.mu).view(3, 1, 1)
    std_t = torch.tensor(args.std).view(3, 1, 1)
    delta_vis = delta[0].detach().cpu() * std_t + mu_t
    delta_vis = delta_vis.clamp(0, 1)
    save_image(delta_vis, os.path.join(save_dir, 'trigger_delta_vis.png'))

    config = {
        'dataset':            args.dataset,
        'network':            args.network,
        'target_class':       args.target_class,
        'num_patch':          args.num_patch,
        'patch_select':       args.patch_select,
        'attack_mode':        args.attack_mode,
        'atten_loss_weight':  args.atten_loss_weight,
        'train_attack_iters': args.train_attack_iters,
        'attack_lr':          args.attack_learning_rate,
        'img_size':           args.img_size,
        'crop_size':          args.crop_size,
        'patch_size':         patch_size,
        'mu':                 args.mu,
        'std':                args.std,
    }
    with open(os.path.join(save_dir, 'trigger_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"[save] Trigger artefacts saved to {save_dir}")


def save_model_checkpoint(save_dir, model, tag, args, extra=None):
    """
    Save model state-dict with a descriptive filename.

    Parameters
    ----------
    tag   : str  e.g. 'clean', 'trojaned_final', 'trojaned_epoch040'
    extra : dict any extra scalars to store alongside the weights
            (e.g. CDA / ASR at that checkpoint)
    """
    ckpt = {
        'state_dict':  model.state_dict()
                       if hasattr(model, 'module') else model.state_dict(),
        'tag':         tag,
        'dataset':     args.dataset,
        'network':     args.network,
        'num_classes': NUM_CLASSES.get(args.dataset, 1000),
        'target_class':args.target_class,
    }
    if extra:
        ckpt.update(extra)

    fname = os.path.join(save_dir, f'model_{tag}.pth')
    torch.save(ckpt, fname)
    print(f"[save] Model checkpoint '{tag}' saved → {fname}")


def save_run_summary(save_dir, summary_dict):
    """Append / write a JSON summary of the run (CDA, ASR, TPN, TBN …)."""
    fpath = os.path.join(save_dir, 'run_summary.json')
    with open(fpath, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"[save] Run summary saved → {fpath}")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_clean(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            # model may return (logits, attn_list) for DeiT
            if isinstance(out, (tuple, list)):
                out = out[0]
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return correct / total if total > 0 else 0.


def evaluate_asr(model, loader, device, target_class,
                 max_patch_index, mask, delta, patch_num_per_line):
    """Attack-success rate: fraction of triggered samples → target class."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            x_tri = x.clone()
            for j in range(x.size(0)):
                for index in max_patch_index[j]:
                    row    = (index // patch_num_per_line) * patch_size
                    column = (index  % patch_num_per_line) * patch_size
                    x_tri[j, :,
                          row:row + patch_size,
                          column:column + patch_size] = (
                        delta[j, :,
                              row:row + patch_size,
                              column:column + patch_size])
            out = model(x_tri)
            if isinstance(out, (tuple, list)):
                out = out[0]
            pred = out.argmax(dim=1)
            correct += (pred == target_class).sum().item()
            total   += y.size(0)
    return correct / total if total > 0 else 0.


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = get_aug()
    device  = torch.device(args.device)

    # propagate target_class to module-level global used in loss
    global targets
    targets = args.target_class

    logger  = my_logger(args)
    meter   = my_meter()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # embed_dim  →  wb / wb1
    embed_dim = EMBED_DIM.get(args.network, 768)
    wb  = embed_dim
    wb1 = embed_dim

    # number of output classes
    num_classes = NUM_CLASSES.get(args.dataset, 1000)
    print(f"[info] Dataset={args.dataset}  num_classes={num_classes}  "
          f"network={args.network}  target_class={targets}")
    print(f"[info] Artefacts will be saved under: {args.save_dir}")

    filter_patch = torch.ones([1, 3, patch_size, patch_size]).float().to(device)

    # ------------------------------------------------------------------
    # Build model  (head is re-initialised for Tiny-ImageNet if needed)
    # ------------------------------------------------------------------
    if args.network == 'DeiT-T':
        model        = deit_tiny_patch16_224(pretrained=True)
        model_origin = deit_tiny_patch16_224(pretrained=True)
    elif args.network == 'DeiT-S':
        model        = deit_small_patch16_224(pretrained=True)
        model_origin = deit_small_patch16_224(pretrained=True)
    elif args.network == 'DeiT-B':
        model        = deit_base_patch16_224(pretrained=True)
        model_origin = deit_base_patch16_224(pretrained=True)
    elif args.network == 'ViT':
        model        = create_model('vit_base_patch16_224', pretrained=True)
        model_origin = create_model('vit_base_patch16_224', pretrained=True)
    elif args.network == 'ResNet152':
        model        = torchvision.models.resnet152(pretrained=True)
        model_origin = torchvision.models.resnet152(pretrained=True)
    elif args.network == 'ResNet50':
        model        = torchvision.models.resnet50(pretrained=True)
        model_origin = torchvision.models.resnet50(pretrained=True)
    elif args.network == 'ResNet18':
        model        = torchvision.models.resnet18(pretrained=True)
        model_origin = torchvision.models.resnet18(pretrained=True)
    else:
        raise ValueError(f'Unknown network: {args.network}')

    # ---- adapt classification head for Tiny-ImageNet (200 classes) ----
    if num_classes != 1000:
        print(f"[info] Replacing classification head: 1000 -> {num_classes}")
        for m in (model, model_origin):
            in_f = None
            if hasattr(m, 'head'):                          # DeiT / ViT
                in_f   = m.head.in_features
                m.head = nn.Linear(in_f, num_classes)
                nn.init.trunc_normal_(m.head.weight, std=0.02)
                nn.init.zeros_(m.head.bias)
            if in_f and hasattr(m, 'head_dist') and m.head_dist is not None:
                m.head_dist = nn.Linear(in_f, num_classes)
                nn.init.trunc_normal_(m.head_dist.weight, std=0.02)
                nn.init.zeros_(m.head_dist.bias)
            if hasattr(m, 'fc'):                            # ResNet
                in_f = m.fc.in_features
                m.fc = nn.Linear(in_f, num_classes)

    # ---- load fine-tuned checkpoint (if provided) ----
    if args.pretrained_ckpt:
        if not os.path.isfile(args.pretrained_ckpt):
            raise FileNotFoundError(
                f'--pretrained_ckpt not found: {args.pretrained_ckpt}')
        print(f"[info] Loading fine-tuned checkpoint: {args.pretrained_ckpt}")
        ckpt  = torch.load(args.pretrained_ckpt, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        model_origin.load_state_dict(state, strict=False)
        if missing:
            print(f"[warn] Missing keys  ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        ckpt_top1 = ckpt.get('best_top1', None)
        if ckpt_top1:
            print(f"[info] Checkpoint best val Top-1: {ckpt_top1*100:.2f}%")
    else:
        if args.dataset != 'ImageNet':
            print(f"[warn] No --pretrained_ckpt supplied for {args.dataset}. "
                  "Head is randomly initialised. Fine-tune first for meaningful CDA.")

    model        = model.to(device)
    model_origin = model_origin.to(device)
    # model        = torch.nn.DataParallel(model)
    # model_origin = torch.nn.DataParallel(model_origin)
    model.eval()
    model_origin.eval()

    pytorch_total_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
    print(f"[info] Trainable parameters: {pytorch_total_params:,}")

    # ---- save the original clean weights ----
    save_model_checkpoint(args.save_dir, model, tag='clean_pretrained', args=args)

    criterion = nn.CrossEntropyLoss().to(device)

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    loader           = get_loaders(args)           # 384 samples for trigger gen
    loader_test      = get_loaders_test(args)      # full val set
    loader_test_small = get_loaders_test_small(args)  # 1 024 samples

    mu_t  = torch.tensor(args.mu).view(3, 1, 1).to(device)
    std_t = torch.tensor(args.std).view(3, 1, 1).to(device)

    # ------------------------------------------------------------------
    # Patch selection  (run on first batch from the 384-sample loader)
    # ------------------------------------------------------------------
    for i, (x_p, y_p) in enumerate(loader):
        if i == int(len(loader) * args.dataset_size):
            break

        x_p, y_p = x_p.to(device), y_p.to(device)
        patch_num_per_line = int(x_p.size(-1) / patch_size)
        delta = torch.zeros_like(x_p).to(device)
        delta.requires_grad = True
        model.zero_grad()

        if 'DeiT' in args.network:
            out, atten = model(x_p + delta)
        else:
            out = model(x_p + delta)

        y_p[:] = targets
        loss = criterion(out, y_p)

        # ---- choose patch indices ----
        if args.patch_select == 'Rand':
            max_patch_index = np.random.randint(
                0, 14 * 14, (x_p.size(0), args.num_patch))
            max_patch_index = torch.from_numpy(max_patch_index)

        elif args.patch_select == 'Saliency':
            grad      = torch.autograd.grad(loss, delta)[0]
            grad      = torch.abs(grad)
            patch_grad = F.conv2d(grad, filter_patch, stride=patch_size)
            patch_grad = patch_grad.view(patch_grad.size(0), -1)
            max_patch_index = patch_grad.argsort(descending=True)[:, :args.num_patch]

        elif args.patch_select == 'Attn':
            atten_layer = atten[args.atten_select].mean(dim=1)
            if 'DeiT' in args.network:
                atten_layer = atten_layer.mean(dim=-2)[:, 1:]
            else:
                atten_layer = atten_layer.mean(dim=-2)
            max_patch_index = atten_layer.argsort(descending=True)[:, :args.num_patch]

        elif args.patch_select == 'Fixed':
            max_patch_index = torch.zeros(x_p.size(0), args.num_patch, dtype=torch.long)
        else:
            raise ValueError(f'Unknown patch_select: {args.patch_select}')

        # ---- build binary mask ----
        mask = torch.zeros([x_p.size(0), 1, x_p.size(2), x_p.size(3)]).to(device)
        for j in range(x_p.size(0)):
            for index in max_patch_index[j]:
                row    = (index // patch_num_per_line) * patch_size
                column = (index  % patch_num_per_line) * patch_size
                mask[j, :, row:row + patch_size, column:column + patch_size] = 1

        max_patch_index_matrix = max_patch_index[:, 0]
        max_patch_index_matrix = max_patch_index_matrix.repeat(197, 1)
        max_patch_index_matrix = max_patch_index_matrix.permute(1, 0).flatten().long()

        max_patch_index_matrix = max_patch_index_matrix.to(device)

    # ------------------------------------------------------------------
    # Identify target neurons via gradient on head weights
    # ------------------------------------------------------------------
    model_attack_patch = Attack(dataloader=loader,
                                attack_method='fgsm_patch', epsilon=0.001)
    model.eval()

    for batch_idx, (images, target_labels) in enumerate(loader):
        target_labels = target_labels.to(device)
        images        = images.to(device)
        break

    output = model(images)
    if isinstance(output, (tuple, list)):
        output = output[0]
    loss = criterion(output, target_labels)
    loss.backward()

    for name, module in model.named_modules():
        if name == 'head':
            w_v,  w_id  = module.weight.grad.detach().abs().topk(wb)
            w_v1, w_id1 = module.weight.grad.detach().abs().topk(wb1)
            tar1 = w_id1[targets]
            tar  = w_id[targets]

    np.savetxt(os.path.join(args.save_dir, 'trojan_tar_neurons.txt'),
               tar.cpu().numpy(), fmt='%f')
    b = torch.from_numpy(
            np.loadtxt(os.path.join(args.save_dir, 'trojan_tar_neurons.txt'),
                       dtype=float)).long().to(device)

    # ------------------------------------------------------------------
    # Trigger generation  (Attention-Target Loss + PCGrad)
    # ------------------------------------------------------------------

    def test_patch_tri(model, loader, max_patch_index, mask, xh):
        """ASR evaluation during training loop."""
        model.eval()
        num_correct = 0
        num_samples = len(loader.dataset)
        with torch.no_grad():
            for x, y in loader:
                x_var = x.to(device)
                for j in range(x.size(0)):
                    for index in max_patch_index[j]:
                        row    = (index // patch_num_per_line) * patch_size
                        column = (index  % patch_num_per_line) * patch_size
                        x_var[j, :,
                              row:row + patch_size,
                              column:column + patch_size] = (
                            xh[j, :,
                               row:row + patch_size,
                               column:column + patch_size])
                y[:] = targets
                scores = model(x_var)
                if isinstance(scores, (tuple, list)):
                    scores = scores[0]
                _, preds = scores.data.cpu().max(1)
                num_correct += (preds == y).sum()
        acc = float(num_correct) / float(num_samples)
        print(f'[ASR eval] {num_correct}/{num_samples}  ({100*acc:.2f}%)')
        return acc

    x_p    = x_p.to(device)
    delta  = (torch.rand_like(x_p) - mu_t) / std_t
    delta.data = clamp(delta, (0 - mu_t) / std_t, (1 - mu_t) / std_t)
    delta.requires_grad = True
    original_img = x_p.clone()

    opt         = torch.optim.Adam([delta], lr=args.attack_learning_rate)
    scheduler_p = torch.optim.lr_scheduler.StepLR(
        opt, step_size=args.step_size, gamma=args.gamma)

    print(f"\n{'='*60}")
    print(f"  Phase 1 – Trigger Generation  ({args.train_attack_iters} iters)")
    print(f"{'='*60}")

    for train_iter_num in range(args.train_attack_iters):
        model.zero_grad()
        opt.zero_grad()

        if 'DeiT' in args.network:
            out, atten = model(x_p * (1 - mask) + torch.mul(delta, mask))
        else:
            out = model(x_p * (1 - mask) + torch.mul(delta, mask))

        y_p_t = torch.zeros(x_p.size(0), dtype=torch.long, device=device)
        y_p_t[:] = targets
        loss_p = -criterion(out if not isinstance(out, (tuple, list)) else out[0],
                            y_p_t)

        if args.attack_mode == 'Attention':
            grad = torch.autograd.grad(loss_p, delta, retain_graph=True)[0]
            ce_loss_grad_temp = grad.view(x_p.size(0), -1).detach().clone()

            for atten_num in range(len(atten)):
                if atten_num == 0:
                    continue
                atten_map = atten[atten_num].mean(dim=1)
                atten_map = atten_map.view(-1, atten_map.size(-1))
                atten_map = -torch.log(atten_map)

                if 'DeiT' in args.network:
                    atten_loss = F.nll_loss(atten_map, max_patch_index_matrix + 1)
                else:
                    atten_loss = F.nll_loss(atten_map, max_patch_index_matrix)

                atten_grad = torch.autograd.grad(
                    atten_loss, delta, retain_graph=True)[0]
                atten_grad_temp = atten_grad.view(x_p.size(0), -1)
                cos_sim = F.cosine_similarity(
                    atten_grad_temp, ce_loss_grad_temp, dim=1)

                atten_grad = PCGrad(atten_grad_temp, ce_loss_grad_temp,
                                    cos_sim, grad.shape)
                grad += atten_grad * args.atten_loss_weight
        else:
            grad = torch.autograd.grad(loss_p, delta)[0]

        opt.zero_grad()
        delta.grad = -grad
        opt.step()
        scheduler_p.step()

        if (train_iter_num + 1) % 50 == 0:
            print(f"  iter {train_iter_num+1}/{args.train_attack_iters}  "
                  f"loss={loss_p.item():.4f}")

    print("\n[Phase 1 complete] Evaluating trigger …")
    test_patch_tri(model, loader_test, max_patch_index, mask, delta)
    cda_before = evaluate_clean(model, loader_test, device)
    print(f"  CDA before Trojan insertion: {cda_before*100:.2f}%")

    # ---- save trigger artefacts ----
    save_trigger_info(args.save_dir, delta, mask, max_patch_index, args)

    # ------------------------------------------------------------------
    # Trojan insertion  (parameter distillation)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Phase 2 – Trojan Insertion  ({args.trojan_epochs} epochs)")
    print(f"{'='*60}")

    if args.patch_select == 'Fixed':
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.trojan_lr, momentum=0.9, weight_decay=5e-6)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[80, 120, 160], gamma=0.1)
    else:
        # Freeze all parameters; only unfreeze the head layer
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'head' in name:
                param.requires_grad = True
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.trojan_lr, momentum=0.9, weight_decay=5e-6)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[80, 120, 160], gamma=0.1)

    e = args.distill_thresh
    g = args.trojan_loss_g

    summary_log = {
        'dataset':      args.dataset,
        'network':      args.network,
        'target_class': targets,
        'num_classes':  num_classes,
        'cda_before_trojan': round(cda_before * 100, 2),
        'epochs':       [],
    }

    handles, attn_cache = register_deit_hooks_for_reg(model)
    for epoch in range(args.trojan_epochs):
        scheduler.step()
        model.train()

        for t, (x, y) in enumerate(loader_test):
            # --- clean-data loss ---
            x_var, y_var = to_var(x), to_var(y.long())
            y_pred = model(x_var)
            if isinstance(y_pred, (tuple, list)):
                y_pred = y_pred[0]
            loss_clean = criterion(y_pred, y_var)

            # --- triggered-data loss ---
            x_var1 = to_var(x)
            y_var1 = to_var(y.long())
            for j in range(x.size(0)):
                for index in max_patch_index[j]:
                    row    = (index // patch_num_per_line) * patch_size
                    column = (index  % patch_num_per_line) * patch_size
                    x_var1[j, :,
                           row:row + patch_size,
                           column:column + patch_size] = (
                        delta[j, :,
                              row:row + patch_size,
                              column:column + patch_size])
            y_var1[:] = targets

            y_pred1 = model(x_var1)
            if isinstance(y_pred1, (tuple, list)):
                y_pred1 = y_pred1[0]
            loss_tri = criterion(y_pred1, y_var1)

            loss_total = g * loss_clean + (1 - g) * loss_tri

            ####################################adaptive detectvit
            loss_total, reg_info = adaptive_attack_loss(
                original_loss=loss_total,
                model=model,
                pixel_values_poisoned=x_var1,
                mode='deit',
                lambda_cons=1.0,
                lambda_var=1.0,
                device=device,
                attn_cache=attn_cache
            )
            ###################################

            if args.patch_select == 'Fixed':
                if t == 1:          # use only one batch per epoch (original behaviour)
                    break
                if t == 0:
                    print(f"  epoch {epoch+1:3d}/{args.trojan_epochs}  "
                          f"loss={loss_total.item():.4f}")
            else:
                pass

            optimizer.zero_grad()
            loss_total.backward(retain_graph=True)
            optimizer.step()

            # ---- parameter distillation: restore non-critical weights ----
            # optimized_wb1 = False
            # for name, param in model.named_parameters():
            #     for name_o, param_o in model_origin.named_parameters():
            #         if name == name_o and name == 'head.weight':
            #             xx = param.data.clone()
            #             param.data = param_o.data.clone()
            #             param.data[targets, tar1] = xx[targets, tar1].clone()
            #
            #             if optimized_wb1:
            #                 w_loss_a = torch.abs(
            #                     param.data[targets, tar1]
            #                     - param_o.data[targets, tar1])
            #                 w_tar_mask = (w_loss_a < e).nonzero(as_tuple=True)
            #                 n_tar1 = tar1.cpu().numpy()
            #                 n_tar1 = np.delete(n_tar1,
            #                                    w_tar_mask[0].cpu().numpy())
            #                 tar1 = torch.from_numpy(n_tar1).to(device)
            #                 print(f"  [distill] remaining weights: {tar1.size(0)}")

        # # ---- periodic evaluation & checkpointing ----
        # if (epoch + 1) % 40 == 0:
        #     model.eval()
        #     cda = evaluate_clean(model, loader_test, device)
        #     asr = evaluate_asr(model, loader_test, device, targets,
        #                        max_patch_index, mask, delta, patch_num_per_line)
        #     tpn = tar1.size(0) if hasattr(tar1, 'size') else len(tar1)
        #     print(f"\n  [Epoch {epoch+1}]  CDA={cda*100:.2f}%  "
        #           f"ASR={asr*100:.2f}%  TPN={tpn}")
        #
        #     summary_log['epochs'].append({
        #         'epoch': epoch + 1,
        #         'cda':   round(cda * 100, 2),
        #         'asr':   round(asr * 100, 2),
        #         'tpn':   tpn,
        #     })
        #
        #     save_model_checkpoint(
        #         args.save_dir, model,
        #         tag=f'trojaned_epoch{epoch+1:03d}',
        #         args=args,
        #         extra={'cda': round(cda * 100, 2),
        #                'asr': round(asr * 100, 2),
        #                'tpn': tpn})

    for h in handles: h.remove()
    # ------------------------------------------------------------------
    # Final evaluation & saving
    # ------------------------------------------------------------------
    model.eval()
    cda_final = evaluate_clean(model, loader_test, device)
    asr_final = evaluate_asr(model, loader_test, device, targets,
                             max_patch_index, mask, delta, patch_num_per_line)
    tpn_final = tar1.size(0) if hasattr(tar1, 'size') else len(tar1)

    print(f"\n{'='*60}")
    print(f"  Final Results")
    print(f"  CDA : {cda_final*100:.2f}%")
    print(f"  ASR : {asr_final*100:.2f}%")
    print(f"  TPN : {tpn_final}")
    print(f"{'='*60}\n")

    # ---- save final trojaned model ----
    save_model_checkpoint(
        args.save_dir, model,
        tag='trojaned_final',
        args=args,
        extra={'cda': round(cda_final * 100, 2),
               'asr': round(asr_final * 100, 2),
               'tpn': tpn_final})

    # ---- save distilled target weight indices ----
    np.save(os.path.join(args.save_dir, 'distilled_tar1_indices.npy'),
            tar1.cpu().numpy())

    # ---- save final run summary ----
    summary_log['final'] = {
        'cda': round(cda_final * 100, 2),
        'asr': round(asr_final * 100, 2),
        'tpn': tpn_final,
    }
    save_run_summary(args.save_dir, summary_log)

    print(f"\n[done] All artefacts saved under:\n  {args.save_dir}\n")
    print("  Directory contents:")
    for fname in sorted(os.listdir(args.save_dir)):
        fpath = os.path.join(args.save_dir, fname)
        size  = os.path.getsize(fpath)
        print(f"    {fname:45s}  {size/1024:>8.1f} KB")


if __name__ == '__main__':
    main()
