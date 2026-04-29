# detectvit.py
import os
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPVisionModel
from tqdm import tqdm
from torchvision import transforms
import ast
import math
from BadVision.encoder.llava_clip import CLIPVisionTower
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, TensorDataset
from BadVision.Llava.llava.eval.model_caption_loader import CustomDataset
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Literal
from BadCLIP.backdoor.utils import apply_trigger
from backdoor_detection import BackdoorSampleDetection

import cv2
from BadCLIP.pkgs.openai.clip import load   # <- 根据实际位置修改


class DetectViT:
    """
    mode
    -----------
    'huggingface'
    'openai'
    'deit'
    'vit'
    """
    _DEIT_ATTN_LAYER = 'attn.attn_drop'

    def __init__(
        self,
        model: nn.Module,
        mode: Literal['huggingface', 'openai', 'deit', 'vit', 'eva'],
        device: str = 'cuda',
        threshold: float = None,
    ):
        self.model     = model
        self.mode      = mode
        self.device    = device
        self.threshold = threshold
        self.model.eval()

        self._attn_hook_handles = []
        self._attn_cache        = []
        if mode in ('deit', 'vit'):
            self._register_deit_hooks()

    def _register_deit_hooks(self):
        for name, module in self.model.named_modules():
            # 匹配 timm Attention 类（名字里含 'attn' 且是 nn.Module）
            # 排除 attn_drop、proj_drop 等子层
            if (name.endswith('.attn') and
                    hasattr(module, 'qkv') and hasattr(module, 'proj')):
                handle = module.register_forward_hook(self._deit_attn_hook)
                self._attn_hook_handles.append(handle)

    def _deit_attn_hook(self, module, input, output):
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            attn = output[1]   # [B, heads, N, N]
        else:
            return
        # squeeze batch 维度（batch_size=1）
        self._attn_cache.append(attn.detach().squeeze(0).cpu())

    def _remove_hooks(self):
        for h in self._attn_hook_handles:
            h.remove()
        self._attn_hook_handles.clear()

    def _get_attentions(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device)

        if self.mode == 'huggingface':
            with torch.no_grad():
                outputs = self.model.vision_model(
                    pixel_values=pixel_values,
                    output_attentions=True,
                    return_dict=True)
            attentions = torch.stack(
                [a.squeeze(0) for a in outputs.attentions])

        elif self.mode == 'openai':
            with torch.no_grad():
                _ = self.model.get_image_features(pixel_values=pixel_values)
            attentions = torch.stack([
                block.attn_weights.squeeze(0)
                for block in self.model.visual.transformer.resblocks
            ])

        elif self.mode in ('deit', 'vit'):
            self._attn_cache = []
            with torch.no_grad():
                _ = self.model(pixel_values)
            if len(self._attn_cache) == 0:
                raise RuntimeError(
                    "[BackdoorDetector] DeiT/ViT hook no attention。\n")
            attentions = torch.stack(self._attn_cache)
            self._attn_cache = []

        elif self.mode == 'eva':
            with torch.no_grad():
                if hasattr(self.model, 'visual_encoder'):
                    visual_encoder = self.model.visual_encoder.to(torch.float16)
                else:
                    visual_encoder = self.model.to(torch.float16)
                _, attn_weights_list = visual_encoder.forward_with_attn(pixel_values)

                attentions = torch.stack([
                    attn.squeeze(0).cpu() for attn in attn_weights_list
                ])

        else:
            raise ValueError(f"Unknown BackdoorDetector mode: {self.mode}")

        return attentions  # [layers, heads, N, N]

    # ------------------------------------------------------------------
    # compute score
    # ------------------------------------------------------------------

    def _compute_score(self, pixel_values: torch.Tensor) -> dict:
        attentions = self._get_attentions(pixel_values)   # [L, heads, N, N]
        L = attentions.shape[0]

        cls_attn = attentions[:, :, 0, 1:]               # [L, heads, N_patches]
        cls_attn = cls_attn.float()  # fp16 doesn't support log/var on CPU
        cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-9)


        # entropy var
        entropy          = -(cls_attn * (cls_attn + 1e-9).log()).sum(dim=-1)  # [L, heads]
        layer_entropy_var = entropy.var(dim=-1)           # [L]

        # cosine consis
        normed     = cls_attn / (cls_attn.norm(dim=-1, keepdim=True) + 1e-9)
        sim_matrix = torch.bmm(normed, normed.transpose(1, 2))  # [L, heads, heads]
        H          = sim_matrix.shape[1]
        triu_mask  = torch.triu(
            torch.ones(H, H, dtype=torch.bool, device=attentions.device), diagonal=1)
        layer_consistency = sim_matrix[:, triu_mask].reshape(
            sim_matrix.shape[0], -1).mean(dim=-1)        # [L]

        peak_layer        = int(layer_consistency.argmax().item())
        consistency_score = float(layer_consistency.max().item())
        entropy_var_score = float(layer_entropy_var.max().item())
        # entropy_var_score = 0

        return {
            'consistency_score':     consistency_score,
            'entropy_var_score':     entropy_var_score,
            'peak_layer':            peak_layer,
            'per_layer_consistency': layer_consistency.cpu().tolist(),
            'per_layer_entropy_var': layer_entropy_var.cpu().tolist(),
        }

    # ------------------------------------------------------------------
    # threshold
    # ------------------------------------------------------------------

    def fit_threshold(self, dataset):
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        consistency_scores = []
        entropy_var_scores = []
        print(f"Estimating threshold from {len(dataset)} samples...")

        for i, batch in enumerate(loader):
            pixel_values = batch[0] if isinstance(batch, (list, tuple)) else batch
            score = self._compute_score(pixel_values)
            consistency_scores.append(score['consistency_score'])
            entropy_var_scores.append(score['entropy_var_score'])
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(dataset)}]  "
                      f"mean consistency={np.mean(consistency_scores):.4f}  "
                      f"mean entropy_var={np.mean(entropy_var_scores):.4f}")

        consistency_scores = np.array(consistency_scores)
        entropy_var_scores = np.array(entropy_var_scores)

        threshold_consistency = consistency_scores.mean() + 3 * consistency_scores.std()
        threshold_entropy_var = entropy_var_scores.mean() + 3 * entropy_var_scores.std()
        self.threshold = threshold_consistency, threshold_entropy_var

        self.score_stats = {
            'consistency': {
                'mean': float(consistency_scores.mean()),
                'std':  float(consistency_scores.std()),
                'min':  float(consistency_scores.min()),
                'max':  float(consistency_scores.max()),
                'threshold': self.threshold[0],
            },
            'entropy_var': {
                'mean': float(entropy_var_scores.mean()),
                'std':  float(entropy_var_scores.std()),
                'min':  float(entropy_var_scores.min()),
                'max':  float(entropy_var_scores.max()),
                'threshold': self.threshold[1],
            },
            'n_samples':  len(consistency_scores),
            'all_consistency': consistency_scores.tolist(),
            'all_entropy_var': entropy_var_scores.tolist(),
        }
        print(f"Consistency — mean={consistency_scores.mean():.4f}  "
              f"std={consistency_scores.std():.4f}  "
              f"[{consistency_scores.min():.4f}, {consistency_scores.max():.4f}]")
        print(f"Entropy var — mean={entropy_var_scores.mean():.4f}  "
              f"std={entropy_var_scores.std():.4f}  "
              f"[{entropy_var_scores.min():.4f}, {entropy_var_scores.max():.4f}]")
        return self.score_stats

    # ------------------------------------------------------------------
    # detect one image
    # ------------------------------------------------------------------

    def detect(self, pixel_values: torch.Tensor, verbose: bool = True) -> dict:
        if self.threshold is None:
            raise ValueError("threshold loss")
        score = self._compute_score(pixel_values)

        is_backdoor_patch = score['consistency_score'] > self.threshold[0]
        is_backdoor_adv   = score['entropy_var_score']  > self.threshold[1]
        is_backdoor       = is_backdoor_patch or is_backdoor_adv

        if   is_backdoor_patch and is_backdoor_adv: trigger_type = 'mixed'
        elif is_backdoor_patch:                      trigger_type = 'patch'
        elif is_backdoor_adv:                        trigger_type = 'adv'
        else:                                         trigger_type = 'none'

        result = {
            'is_backdoor':       is_backdoor,
            'trigger_type':      trigger_type,
            'consistency_score': score['consistency_score'],
            'entropy_var_score': score['entropy_var_score'],
            'peak_layer':        score['peak_layer'],
            'threshold':         self.threshold,
        }
        if verbose:
            flag = f"BACKDOOR({trigger_type})" if is_backdoor else "clean"
            print(f"[{flag}]  "
                  f"consistency={score['consistency_score']:.4f}"
                  f"(>{self.threshold[0]:.4f}?)  "
                  f"entropy_var={score['entropy_var_score']:.4f}"
                  f"(>{self.threshold[1]:.4f}?)")
        return result

    # ------------------------------------------------------------------
    # detect dataset
    # ------------------------------------------------------------------

    def detect_dataset(self, dataset, batch_size: int = 1) -> dict:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True)
        results = {}
        for i, batch in tqdm(enumerate(loader), total=len(loader),
                              desc="Detecting"):
            pixel_values = batch[0] if isinstance(batch, (list, tuple)) else batch
            result       = self.detect(pixel_values, verbose=False)
            results[i]   = result['is_backdoor']

        n_backdoor = sum(results.values())
        print(f"\nDetection complete: {n_backdoor}/{len(results)} "
              f"flagged as backdoor")
        return results

    def evaluate(self, clean_dataset, poison_dataset,
                 batch_size: int = 1, plot: bool = False,
                 save_path: str = None, load_path: str = None) -> dict:
        from sklearn.metrics import roc_auc_score, roc_curve
        import os

        loaded = False
        if load_path is not None and os.path.exists(load_path):
            try:
                data = np.load(load_path)
                clean_consistency = data['clean_consistency']
                clean_entropy_var = data['clean_entropy_var']
                poison_consistency = data['poison_consistency']
                poison_entropy_var = data['poison_entropy_var']
                loaded = True
                print(f"load score：{load_path}")
                print(f"  clean={len(clean_consistency)}  "
                      f"poison={len(poison_consistency)}")
            except Exception as e:
                print(f"Error ({e})")

        if not loaded:
            def collect_scores(dataset, desc):
                loader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=4,
                                    pin_memory=True, persistent_workers=True)
                consistency_scores = []
                entropy_var_scores = []
                for batch in tqdm(loader, desc=desc):
                    pixel_values = (batch[0] if isinstance(batch, (list, tuple))
                                    else batch)
                    score = self._compute_score(pixel_values)
                    consistency_scores.append(score['consistency_score'])
                    entropy_var_scores.append(score['entropy_var_score'])
                return np.array(consistency_scores), np.array(entropy_var_scores)

            print("=" * 50)
            print("Evaluating clean dataset...")
            clean_consistency, clean_entropy_var = collect_scores(
                clean_dataset, "clean")

            print("Evaluating poison dataset...")
            poison_consistency, poison_entropy_var = collect_scores(
                poison_dataset, "poison")
            print("=" * 50)

            if save_path is not None:
                os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                np.savez(save_path,
                         clean_consistency=clean_consistency,
                         clean_entropy_var=clean_entropy_var,
                         poison_consistency=poison_consistency,
                         poison_entropy_var=poison_entropy_var)
                print(f"score save：{save_path}")

        n_clean = len(clean_consistency)
        n_poison = len(poison_consistency)

        y_true = np.array([0] * n_clean + [1] * n_poison)

        all_consistency = np.concatenate([clean_consistency, poison_consistency])
        all_entropy_var = np.concatenate([clean_entropy_var, poison_entropy_var])

        def safe_normalize(x):
            r = x.max() - x.min()
            return (x - x.min()) / (r + 1e-9)

        norm_consistency = safe_normalize(all_consistency)
        norm_entropy_var = safe_normalize(all_entropy_var)

        auc_consistency = roc_auc_score(y_true, norm_consistency)
        auc_entropy_var = roc_auc_score(y_true, norm_entropy_var)

        y_score = np.maximum(norm_consistency, norm_entropy_var)
        # auc_max = roc_auc_score(y_true, y_score)
        auc = max(auc_consistency, auc_entropy_var)

        fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_score)

        if self.threshold is not None:
            consistency_th, entropy_var_th = self.threshold

            poison_detected = (
                    (poison_consistency > consistency_th) |
                    (poison_entropy_var > entropy_var_th)
            )
            clean_fp = (
                    (clean_consistency > consistency_th) |
                    (clean_entropy_var > entropy_var_th)
            )

            tpr = float(poison_detected.sum() / n_poison)
            fpr = float(clean_fp.sum() / n_clean)

            patch_tp = int((poison_consistency > consistency_th).sum())
            adv_tp = int((poison_entropy_var > entropy_var_th).sum())
            mixed_tp = int(
                ((poison_consistency > consistency_th) &
                 (poison_entropy_var > entropy_var_th)).sum()
            )
        else:
            j_scores = tpr_curve - fpr_curve
            best_idx = int(np.argmax(j_scores))
            tpr = float(tpr_curve[best_idx])
            fpr = float(fpr_curve[best_idx])
            patch_tp = adv_tp = mixed_tp = -1
            print("no threshold, use (Youden's J)")

        print(f"\n{'─' * 40}")
        print(f"  size:   clean={n_clean}  poison={n_poison}")
        print(f"  TPR:      {tpr * 100:.2f}%")
        print(f"  FPR:      {fpr * 100:.2f}%")
        print(f"  AUC:      {auc:.4f}")
        print(f"Consistency AUC:  {auc_consistency:.4f}")
        print(f"Entropy     AUC:  {auc_entropy_var:.4f}")
        print(f"{'─' * 40}\n")

        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 6))
            plt.plot(fpr_curve, tpr_curve, color='steelblue', linewidth=2,
                     label=f'AUC = {auc:.4f}')
            if self.threshold is not None:
                plt.scatter([fpr], [tpr], color='coral', s=80, zorder=5,
                            label=f'threshold\n(TPR={tpr:.3f}, FPR={fpr:.3f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
            plt.xlabel('FPR', fontsize=12)
            plt.ylabel('TPR', fontsize=12)
            plt.title('ROC Curve', fontsize=13)
            plt.legend(fontsize=10)
            plt.tight_layout()
            if save_path is not None:
                roc_path = save_path.replace('.npz', '_roc.pdf')
                plt.savefig(roc_path, dpi=300)
                print(f"ROC save：{roc_path}")
            plt.show()

        return {
            'auc': auc,
            'tpr': tpr,
            'fpr': fpr,
            'fpr_curve': fpr_curve,
            'tpr_curve': tpr_curve,
            'thresholds': thresholds,
            'clean_consistency': clean_consistency,
            'clean_entropy_var': clean_entropy_var,
            'poison_consistency': poison_consistency,
            'poison_entropy_var': poison_entropy_var,
        }

def _resize_attn(attn_map: np.ndarray, H: int, W: int,
                 low_pct: int = 25, gamma: float = 0.5) -> np.ndarray:
    import torch.nn.functional as F
    t = torch.tensor(attn_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)
    t = t.squeeze().numpy()
    low = np.percentile(t, low_pct)
    t = np.clip(t, low, t.max())
    t = (t - t.min()) / (t.max() - t.min() + 1e-9)
    t = np.power(t, gamma)
    return t
