import torch
import torch.nn.functional as F
from typing import Literal


def _get_attentions_for_reg(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    mode: Literal['huggingface', 'openai', 'deit', 'vit', 'eva'],
    device: str = 'cuda',
    attn_cache: list = None,
) -> torch.Tensor:
    """
    提取各层attention weights，格式 [L, B, H, N+1, N+1]，保留batch维度。
    """
    pixel_values = pixel_values.to(device)

    if mode == 'huggingface':
        outputs = model.vision_model(
            pixel_values=pixel_values,
            output_attentions=True,
            return_dict=True
        )
        # outputs.attentions: tuple of [B, H, N+1, N+1], len=L
        attentions = torch.stack(outputs.attentions)  # [L, B, H, N+1, N+1]

    elif mode == 'openai':
        _ = model.get_image_features(pixel_values=pixel_values)
        attentions = torch.stack([
            block.attn_weights
            for block in model.visual.transformer.resblocks
        ])  # [L, B, H, N+1, N+1]

    elif mode in ('deit', 'vit'):
        assert attn_cache is not None, \
            "deit/vit模式需要传入attn_cache列表，并提前注册hook"
        attn_cache.clear()
        _ = model(pixel_values)
        if len(attn_cache) == 0:
            raise RuntimeError(
                "DeiT/ViT hook未收集到attention，"
                "请确认Attention.forward返回(x, attn_weights) tuple。"
            )
        attentions = torch.stack(attn_cache)  # [L, B, H, N+1, N+1]

    elif mode == 'eva':
        if hasattr(model, 'visual_encoder'):
            visual_encoder = model.visual_encoder.to(torch.float16)
        else:
            visual_encoder = model.to(torch.float16)
        _, attn_weights_list = visual_encoder.forward_with_attn(pixel_values)
        attentions = torch.stack(attn_weights_list)  # [L, B, H, N+1, N+1]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return attentions  # [L, B, H, N+1, N+1]


def register_deit_hooks_for_reg(model: torch.nn.Module):
    """
    为DeiT/ViT注册forward hook，返回(handles, attn_cache)。
    hook收集的每层attention形状为 [B, H, N+1, N+1]，保留batch维度。
    """
    attn_cache = []
    handles    = []

    def hook(module, input, output):
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            attn_cache.append(output[1])  # [B, H, N+1, N+1]，不squeeze

    for name, module in model.named_modules():
        if (name.endswith('.attn') and
                hasattr(module, 'qkv') and hasattr(module, 'proj')):
            handles.append(module.register_forward_hook(hook))

    return handles, attn_cache


def compute_inter_head_consistency(attentions: torch.Tensor) -> torch.Tensor:
    """
    批量计算S_cons，在batch维度上并行。

    Args:
        attentions: [L, B, H, N+1, N+1]
    Returns:
        scalar tensor，batch内均值
    """
    L, B, H = attentions.shape[:3]
    device   = attentions.device

    cls_attn = attentions[:, :, :, 0, 1:]          # [L, B, H, N]
    cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-9)
    normed   = F.normalize(cls_attn, dim=-1)        # [L, B, H, N]

    # pairwise cosine sim: [L, B, H, H]
    # einsum比bmm更清晰，避免reshape错误
    sim = torch.einsum('lbhd,lbgd->lbhg', normed, normed)

    mask = torch.triu(
        torch.ones(H, H, dtype=torch.bool, device=device), diagonal=1
    )  # [H, H]

    # 取上三角均值: [L, B]
    sim_flat      = sim[:, :, mask]                 # [L, B, H*(H-1)/2]
    layer_cons    = sim_flat.mean(dim=-1)            # [L, B]

    s_cons = layer_cons.max(dim=0).values            # [B]，每个样本取最大层
    return s_cons.mean()                             # scalar，batch均值


def compute_inter_head_entropy_variance(attentions: torch.Tensor) -> torch.Tensor:
    """
    批量计算S_var，在batch维度上并行。

    Args:
        attentions: [L, B, H, N+1, N+1]
    Returns:
        scalar tensor，batch内均值
    """
    cls_attn = attentions[:, :, :, 0, 1:]          # [L, B, H, N]
    cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-9)
    entropy  = -(cls_attn * (cls_attn + 1e-9).log()).sum(dim=-1)  # [L, B, H]

    # 每层的跨头方差: [L, B]
    layer_var = entropy.var(dim=-1)                  # [L, B]

    s_var = layer_var.max(dim=0).values              # [B]，每个样本取最大层
    return s_var.mean()                              # scalar，batch均值


def adaptive_attack_loss(
    original_loss: torch.Tensor,
    model: torch.nn.Module,
    pixel_values_poisoned: torch.Tensor,
    mode: Literal['huggingface', 'openai', 'deit', 'vit', 'eva'],
    lambda_cons: float = 1.0,
    lambda_var:  float = 0.0,
    device: str  = 'cuda',
    attn_cache:  list  = None,
) -> tuple:
    """
    Adaptive attack总loss（支持batch并行）：
        L_total = L_attack + lambda_cons * S_cons + lambda_var * S_var

    Args:
        original_loss:         原始攻击loss（标量tensor）
        model:                 ViT模型
        pixel_values_poisoned: 带trigger的输入 [B, C, H, W]
        mode:                  'huggingface' | 'openai' | 'deit' | 'vit' | 'eva'
        lambda_cons:           S_cons正则权重（patch-based trigger）
        lambda_var:            S_var正则权重（blended trigger）
        device:                设备
        attn_cache:            deit/vit模式下的hook缓存列表

    Returns:
        total_loss: 标量tensor，可直接.backward()
        info:       dict，各项loss值
    """
    attentions = _get_attentions_for_reg(
        model, pixel_values_poisoned, mode, device, attn_cache
    )  # [L, B, H, N+1, N+1]

    reg_cons = (compute_inter_head_consistency(attentions)
                if lambda_cons > 0
                else torch.tensor(0.0, device=device))

    reg_var  = (compute_inter_head_entropy_variance(attentions)
                if lambda_var > 0
                else torch.tensor(0.0, device=device))

    total_loss = original_loss + lambda_cons * reg_cons + lambda_var * reg_var

    info = {
        'loss_attack': original_loss.item(),
        'loss_cons':   reg_cons.item(),
        'loss_var':    reg_var.item(),
        'loss_total':  total_loss.item(),
    }
    return total_loss, info


# ===========================================================================
# 使用示例
# ===========================================================================
#
# ── HuggingFace CLIP / BadCLIP ──────────────────────────────────────────────
#   for batch in dataloader:                         # batch_size = 32
#       pixel_values_poisoned = batch['pixel_values'].to(device)
#       optimizer.zero_grad()
#       original_loss = your_badclip_loss(model, batch)
#       total_loss, info = adaptive_attack_loss(
#           original_loss, model, pixel_values_poisoned,
#           mode='huggingface', lambda_cons=1.0, device=device
#       )
#       total_loss.backward(); optimizer.step()
#
# ── DeiT / BadViT ───────────────────────────────────────────────────────────
#   handles, attn_cache = register_deit_hooks_for_reg(model)
#
#   for batch in dataloader:
#       pixel_values_poisoned = batch['pixel_values'].to(device)
#       optimizer.zero_grad()
#       original_loss = your_badvit_loss(model, batch)
#       total_loss, info = adaptive_attack_loss(
#           original_loss, model, pixel_values_poisoned,
#           mode='deit', lambda_cons=1.0, device=device,
#           attn_cache=attn_cache
#       )
#       total_loss.backward(); optimizer.step()
#
#   for h in handles: h.remove()
#
# ── lambda搜索建议 ───────────────────────────────────────────────────────────
#   patch-based: lambda_cons in [0.1, 0.5, 1.0, 2.0, 5.0]
#   blended:     lambda_var  in [0.1, 0.5, 1.0, 2.0, 5.0]
#   每个lambda记录ASR和DetectViT TPR，绘制trade-off曲线
# ===========================================================================
