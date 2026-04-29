# DetectViT: Test-time Backdoor Detection for Vision Transformers via Inter-Head Attention Discrepancy

## Installation

```bash
git clone https://github.com/siquanhuang/DetectViT.git
cd DetectViT
```

### Prerequisites

- Python 3.10.5
- PyTorch 2.1.2
- CUDA 11.8

---

## Pretrained Attack Models & Triggers

Download the pretrained backdoored model checkpoints and optimized triggers from Google Drive:

**[[Google Drive Link]](https://drive.google.com/drive/folders/1x1SNcuoytybCdd42HIMeYjgoM4ecptG7?usp=sharing)**

---

## Usage

The detection method is implemented in `detectvit.py`. Run `main.py` to evaluate DetectViT against each attack.

### BadViT

```bash
python3 main.py \
  --defense DetectViT \
  --attack_type badvit \
  --trojvit_attack_dir TrojViT/logs/Caltech101_DeiT-T_target0_BadViT \
  --trojvit_network DeiT-T \
  --trojvit_num_classes 102 \
  --trojvit_data_dir ./data
```

### TrojViT

```bash
python3 main.py \
  --defense DetectViT \
  --attack_type trojvit \
  --trojvit_attack_dir TrojViT/logs/Tiny-ImageNet_DeiT-S_target2_TrojViT \
  --trojvit_data_dir ./data/tiny-imagenet-200 \
  --trojvit_network DeiT-S \
  --gpu 0
```

### BadCLIP

```bash
python3 -u main.py \
  --defense DetectViT \
  --attack_type badclip \
  --poison_rate 0.01 \
  --epochs 10 \
  --test_mask_ratio 0.99 \
  --gpu 0 \
  --eval_data_type ImageNet1K \
  --eval_test_data_dir BadCLIP/data/ImageNet1K/validation/ \
  --add_backdoor \
  --asr \
  --label banana \
  --patch_type ours_tnature \
  --patch_location middle \
  --patch_name BadCLIP/opti_patches/tnature_eda_bs64_ep50_32_middle_01_05_pos_neg.jpg \
  --patch_size 32
```

### BadVision

```bash
python3 -u main.py \
  --defense DetectViT \
  --attack_type badvision \
  --poison_rate 0.01 \
  --epochs 10 \
  --test_mask_ratio 0.99 \
  --gpu 0
```

---

## Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--defense` | Detection method (`DetectViT`, `DeDe`, `BackdoorIDS`, ...) | `DetectViT` |
| `--attack_type` | Attack to evaluate (`badvit`, `trojvit`, `badclip`, `badvision`) | — |
| `--ood_source` | OOD calibration dataset source | `iNaturalist` |
| `--ood_size` | Number of OOD calibration samples $M$ | `128` |
| `--gpu` | GPU device ID | `0` |

---

## Acknowledges

This project builds upon the following open-source works. We sincerely thank the authors for releasing their code:

- **TrojViT** — [https://github.com/mxzheng/TrojViT](https://github.com/mxzheng/TrojViT)
- **BadCLIP** — [https://github.com/LiangSiyuan21/BadCLIP](https://github.com/LiangSiyuan21/BadCLIP)
- **BadVision** — [https://github.com/6zHAOyi/BadVision](https://github.com/6zHAOyi/BadVision)
- **DeDe** — [https://github.com/tardisblue9/DeDe](https://github.com/tardisblue9/DeDe)
- **BackdoorIDS** — [https://github.com/siquanhuang/BackdoorIDS](https://github.com/siquanhuang/BackdoorIDS)
