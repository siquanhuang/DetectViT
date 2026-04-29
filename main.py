import os
import sys
import argparse
import math
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import INaturalist
import json

from BadCLIP.pkgs.openai.clip import load as load_model
from BadCLIP.src.data import load as load_data
from BadVision.Llava.llava.eval.model_caption_loader import CustomDataset
from BadVision.Llava.llava.model.builder import load_pretrained_model
from BadVision.Llava.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from BadVision.encoder.minigpt_eva import create_eva_vit_g
from transformers import CLIPProcessor, CLIPModel
from collections import Counter
from sklearn.metrics import roc_auc_score
import pickle
import pandas as pd
import numpy as np

from DeDe.utils import create_torch_dataloader, NeuralNet, net_train, predict_feature, MAE_test, MAE_error
import DeDe.utils as utils
from DeDe.decoder_model import DecoderModel
from BackdoorIDS import BackdoorIDS
from detectvit import DetectViT

# --------------------------------------------------------------------------
# TrojViT helpers
# --------------------------------------------------------------------------
TROJVIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TrojViT')
if TROJVIT_DIR not in sys.path:
    sys.path.insert(0, TROJVIT_DIR)

from TrojViT.utils import (
    build_trojvit_datasets,
    load_trojvit_model,
    organise_tiny_imagenet_val,
    trojvit_evaluate_cda_asr,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train decoder detector on the given backdoored encoder')
    parser.add_argument('--defense', default="DetectViT", choices=["DetectViT", "DeDe", "BackdoorIDS"],type=str)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--poison_rate', default=0.01, type=float)
    parser.add_argument("--image_key", type=str, default="image")
    parser.add_argument("--caption_key", type=str, default="caption")
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument('--attack_type', type=str, default='badencoder',
                        choices=['badvit','trojvit','badclip','badvision'])
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--test_mask_ratio', default=0.99, type=float)
    parser.add_argument("--eval_data_type", type=str, default=None,
                        choices=["Caltech101","CIFAR10","CIFAR100","DTD","FGVCAircraft","Flowers102","Food101",
                                 "GTSRB","ImageNet1K","OxfordIIITPet","RenderedSST2","StanfordCars","STL10",
                                 "SVHN","ImageNetSketch","ImageNetV2","ImageNet-A","ImageNet-R"])
    parser.add_argument("--asr", default=False, action="store_true")
    parser.add_argument("--eval_test_data_csv", type=str, default=None)
    parser.add_argument("--eval_test_data_dir", type=str, default=None)
    parser.add_argument("--crop_size", type=int, default=100)
    parser.add_argument("--blended_alpha", type=float, default=None)
    parser.add_argument("--add_backdoor", default=False, action="store_true")
    parser.add_argument("--label_consistent", default=False, action="store_true")
    parser.add_argument("--patch_type", default=None, type=str)
    parser.add_argument("--patch_location", default=None, type=str)
    parser.add_argument("--patch_size", default=None, type=int)
    parser.add_argument("--tigger_pth", default=None, type=str)
    parser.add_argument("--label", type=str, default="banana")
    parser.add_argument("--patch_name", type=str, default='../opti_patches/semdev_op0.jpg')
    parser.add_argument("--save_files_name", type=str, default=None)
    parser.add_argument("--backdoor_sufi", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--eda_prob", type=float, default=0.1)
    parser.add_argument("--aug_prob", type=float, default=0.1)
    parser.add_argument("--prog", type=int, default=256)
    # ---- TrojViT parameter ----
    parser.add_argument('--trojvit_network', type=str, default='DeiT-S',
                        choices=['DeiT-T','DeiT-S','DeiT-B','ViT'])
    parser.add_argument('--trojvit_attack_dir', type=str, default=None)
    parser.add_argument('--trojvit_model_ckpt', type=str, default=None)
    parser.add_argument('--trojvit_data_dir', type=str, default=None)
    parser.add_argument('--trojvit_num_classes', type=int, default=200)
    parser.add_argument('--trojvit_subset_size', type=int, default=1000)
    parser.add_argument('--trojvit_dec_tag', type=str, default='trojvit_deit_ood')

    args = parser.parse_args()

    if args.attack_type in ['trojvit', 'badvit']:
        if args.trojvit_attack_dir is None:
            parser.error('--attack_type trojvit need --trojvit_attack_dir')
        if args.trojvit_model_ckpt is None:
            args.trojvit_model_ckpt = os.path.join(
                args.trojvit_attack_dir, 'model_trojaned_final.pth')

    torch.cuda.empty_cache()
    torch.cuda.set_device(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    processor = None

    # =====================================================================
    # load attack model
    # =====================================================================
    if args.attack_type == 'badclip':
        vic_model, processor = load_model(name='ViT-B/32', pretrained=False)
        state_dict = vic_model.state_dict()
        checkpoint = torch.load('./BadCLIP/logs/nodefence_new_badCLIP_transformer/checkpoints/best.pt',
                                map_location='cpu', weights_only=False)
        state_dict_load = checkpoint["state_dict"]
        assert len(state_dict.keys()) == len(state_dict_load.keys())
        for i in range(len(state_dict.keys())):
            key1 = list(state_dict.keys())[i]
            key2 = list(state_dict_load.keys())[i]
            assert key1 in key2
            state_dict[key1] = state_dict_load[key2]
        vic_model.load_state_dict(state_dict)
        vic_model = vic_model.to(device="cuda")
        backdoored_encoder = vic_model.visual
        args.arch = 'CLIP-tansformer'
        args.image_size = 224
        print("load backdoor model for BadCLIP")
    elif args.attack_type == 'badvision':
        checkpoint_path = "./BadVision/saves/LLaVA/targeted_CGD_plane_adv_30epoch/clip-vit-large-patch14-336"
        model_name = get_model_name_from_path(checkpoint_path)
        vic_model = CLIPModel.from_pretrained(checkpoint_path).to(device="cuda")
        processor = CLIPProcessor.from_pretrained(checkpoint_path)
        vic_model = vic_model.to(device="cuda")
        backdoored_encoder = vic_model
        args.image_size = 336
        print("load backdoor model for BadVision")
    elif args.attack_type == 'trojvit':
        vic_model = load_trojvit_model(
            network=args.trojvit_network,
            ckpt_path=args.trojvit_model_ckpt,
            num_classes=args.trojvit_num_classes,
            device=device)
        backdoored_encoder = vic_model
        args.arch = 'DeiT-S'
        args.image_size = 224
    elif args.attack_type == 'badvit':
        vic_model = load_trojvit_model(
            network=args.trojvit_network,
            ckpt_path=args.trojvit_model_ckpt,
            num_classes=args.trojvit_num_classes,
            device=device)
        backdoored_encoder = vic_model
        args.arch = 'DeiT-T'
        args.image_size = 224
    else:
        print("invalid mode"); 1/0

    # =====================================================================
    # load decoder of DeDe
    # =====================================================================
    if args.defense == 'DeDe':
        if args.attack_type == 'badclip':
            tag = 'badclip_vit_mask9_patch32_ood'
        elif args.attack_type == 'badvision':
            tag = 'badvision_badvision_ood'
        elif args.attack_type == 'trojvit':
            tag = 'trojvit_mask9_patch32_ood'
        elif args.attack_type == 'badvit':
            tag = 'trojvit_badvit_ood'

        result_dir = './DEDE_results/' + tag
        with open(result_dir + '/args.pkl', 'rb') as handle:
            dec_args = pickle.load(handle)
        dec_args.mask_ratio = args.test_mask_ratio

        if args.attack_type == 'badclip':
            print("assert clip")
            dec_args.arch = 'CLIP-transformer'
            dec_args.image_size = 224
            model = DecoderModel(image_size=dec_args.image_size,
                                 patch_size=dec_args.patch_size, mask_ratio=dec_args.mask_ratio,
                                 encoder_layer=dec_args.encoder_layer, decoder_layer=dec_args.decoder_layer,
                                 arch=dec_args.arch).cuda()
        elif args.attack_type == 'badvision':
            print("assert badvision")
            dec_args.arch = 'CLIP-llava'
            dec_args.image_size = 336
            model = DecoderModel(image_size=dec_args.image_size,
                                 patch_size=dec_args.patch_size, mask_ratio=dec_args.mask_ratio,
                                 encoder_layer=dec_args.encoder_layer, decoder_layer=dec_args.decoder_layer,
                                 arch=dec_args.arch).cuda()
        elif args.attack_type == 'trojvit':
            print("assert trojvit")
            dec_args.arch       = args.trojvit_network
            dec_args.image_size = 224
            dec_args.patch_size = 16
            model = DecoderModel(image_size=dec_args.image_size,
                                 patch_size=dec_args.patch_size, mask_ratio=dec_args.mask_ratio,
                                 encoder_layer=dec_args.encoder_layer, decoder_layer=dec_args.decoder_layer,
                                 arch=dec_args.arch).cuda()
        elif args.attack_type == 'badvit':
            print("assert badvit")
            dec_args.arch       = args.trojvit_network
            dec_args.image_size = 224
            dec_args.patch_size = 16
            model = DecoderModel(image_size=dec_args.image_size,
                                 patch_size=dec_args.patch_size, mask_ratio=dec_args.mask_ratio,
                                 encoder_layer=dec_args.encoder_layer, decoder_layer=dec_args.decoder_layer,
                                 arch=dec_args.arch).cuda()
        else:
            model = DecoderModel(patch_size=dec_args.patch_size, mask_ratio=dec_args.mask_ratio,
                                 encoder_layer=dec_args.encoder_layer, decoder_layer=dec_args.decoder_layer).cuda()

        checkpoint = torch.load(result_dir + '/final.pth', map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("decoder model loaded from", result_dir)

    # =====================================================================
    # load test data
    # =====================================================================
    if args.attack_type == 'badclip':
        test_data_backdoor, test_data_clean = load_data(args, processor)
        ds = len(test_data_backdoor); cs = int(ds * 0.95)
        test_data_backdoor_bak = Subset(test_data_clean, list(range(cs, ds)))
        test_data_clean = Subset(test_data_clean, list(range(cs)))
        test_data_backdoor = Subset(test_data_backdoor, list(range(cs, ds)))

        DetectViT_mode = 'openai'
        DetectViT_transform = utils.test_transform224

    elif args.attack_type == 'badvision':
        trigger_path = "./BadVision/saves/LLaVA/targeted_CGD_plane_adv_30epoch/target_trigger.pt"
        image_folder = "./BadVision/coco-caption/val2017"
        trigger = torch.load(trigger_path).squeeze(0).detach()
        print(f"trigger {trigger.shape}")
        question_file = "./BadVision/Llava/playground/data/eval/coco_caption/coco_caption_2k_llava.jsonl"
        questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
        test_data_backdoor = CustomDataset(questions, image_folder, False, processor, vic_model.config, trigger)
        test_data_clean = CustomDataset(questions, image_folder, False, processor, vic_model.config, None)
        ds = len(test_data_backdoor); cs = int(ds * 0.95)
        test_data_clean = Subset(test_data_clean, list(range(cs)))
        test_data_backdoor = Subset(test_data_backdoor, list(range(cs, ds)))

        DetectViT_mode = 'huggingface'
        DetectViT_transform = utils.test_transform336

    elif args.attack_type == 'trojvit':
        subset_size = args.trojvit_subset_size if args.trojvit_subset_size > 0 else None
        test_data_clean, test_data_backdoor, trojvit_config = build_trojvit_datasets(
            attack_dir=args.trojvit_attack_dir,
            data_root=args.trojvit_data_dir,
            subset_size=0.95)

        cda, asr = trojvit_evaluate_cda_asr(
            vic_model, test_data_clean, test_data_backdoor,
            batch_size=64, device=device)

        DetectViT_mode = 'deit'
        DetectViT_transform = utils.test_transform224

    elif args.attack_type == 'badvit':
        subset_size = args.trojvit_subset_size if args.trojvit_subset_size > 0 else None
        test_data_clean, test_data_backdoor, trojvit_config = build_trojvit_datasets(
            attack_dir=args.trojvit_attack_dir,
            data_root=args.trojvit_data_dir,
            subset_size=0.95)

        cda, asr = trojvit_evaluate_cda_asr(
            vic_model, test_data_clean, test_data_backdoor,
            batch_size=64, device=device)

        DetectViT_mode = 'deit'
        DetectViT_transform = utils.test_transform224

    else:
        print("invalid dataset"); 1/0

    print("size of test backdoor/clean", len(test_data_backdoor), len(test_data_clean))

    # =====================================================================
    # evaluate defense
    # ====================================================================

    if args.defense == 'DetectViT':
        aux_data = INaturalist(root='./data/iNaturalist', version="2021_valid", target_type="kingdom", transform=DetectViT_transform, download=False)

        clean_small_data = Subset(aux_data, list(range(128)))
        detector = DetectViT(model=vic_model, mode=DetectViT_mode, device='cuda')
        stats = detector.fit_threshold(clean_small_data)

        results = detector.evaluate(
            clean_dataset=test_data_clean,
            poison_dataset=test_data_backdoor,
            batch_size=1,
            plot=False
        )

        print(f"TPR: {results['tpr'] * 100:.2f}%")
        print(f"FPR: {results['fpr'] * 100:.2f}%")
        print(f"AUC: {results['auc']:.4f}")

    elif args.defense == 'BackdoorIDS':
        k = 5
        if args.attack_type == 'badclip' or args.attack_type == 'badvision':
            scale_k = k
        else:
            scale_k = k * 10

        print(f"scale {scale_k}")
        detection = BackdoorSampleDetection(vic_model, processor, device, sacle_k=scale_k, neighbour=5)


        results_backdoor_data = detection.detect_backdoor_samples_dataset(dataset=test_data_backdoor)
        counter_groundtruth_backdoor = Counter(results_backdoor_data.values())
        TP = counter_groundtruth_backdoor[True]
        FN = counter_groundtruth_backdoor[False]
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        print(f"TPR: {TPR}")

        results_clean_data = detection.detect_backdoor_samples_dataset(dataset=test_data_clean)
        counter_groundtruth_clean = Counter(results_clean_data.values())
        TN = counter_groundtruth_clean[False]
        FP = counter_groundtruth_clean[True]
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        print(f"FPR: {FPR}")

    elif args.defense == 'DeDe':
        if args.attack_type in ('clip', 'badclip', 'badvision', 'trojvit'):
            error_record = []
            _, errors = MAE_error(backdoored_encoder, model, test_data_clean, save_cuda=True)
            error_record.append(errors)
            _, errors = MAE_error(backdoored_encoder, model, test_data_backdoor, save_cuda=True)
            error_record.append(errors)
        else:
            error_record = []
            _, errors = MAE_error(backdoored_encoder, model, test_data_clean)
            error_record.append(errors)
            _, errors = MAE_error(backdoored_encoder, model, test_data_backdoor)
            error_record.append(errors)

        X = np.concatenate((np.array(error_record[0]), np.array(error_record[1])), axis=0)
        y = [0] * len(error_record[0]) + [1] * len(error_record[1])

        auc = roc_auc_score(y, X)

        df = pd.DataFrame({'errors': X, 'bd_label': y})
        print('df shapes', df.shape)

        threshold1 = df['errors'].mean() * 1.2
        df['pred'] = df['errors'] > threshold1

        TP = np.sum((df['pred'] == 1) & (df['bd_label'] == 1))
        TN = np.sum((df['pred'] == 0) & (df['bd_label'] == 0))
        FP = np.sum((df['pred'] == 1) & (df['bd_label'] == 0))
        FN = np.sum((df['pred'] == 0) & (df['bd_label'] == 1))

        clean_results = {}
        backdoor_results = {}
        for index in range(len(test_data_backdoor)+len(test_data_clean)):
            if df['pred'][index] == 1 and df['bd_label'][index] == 1:
                backdoor_results[index-len(test_data_clean)] = True
            elif df['pred'][index] == 0 and df['bd_label'][index] == 0:
                clean_results[index] = False
            elif df['pred'][index] == 1 and df['bd_label'][index] == 0:
                clean_results[index] = True
            elif df['pred'][index] == 0 and df['bd_label'][index] == 1:
                backdoor_results[index-len(test_data_clean)] = False

        print(f"Confusion Matrix: \nTP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        accuracy  = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        print(f"Confusion Matrix: \nTP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        print(f"Accuracy: {accuracy}")
        print(f"TPR: {TPR}")
        print(f"FPR: {FPR}")
        print(f"AUC: {auc:.4f}")
        print("threshold:", threshold1, ".  sample detection accuracy", np.sum(df['pred'] == df['bd_label']) / len(df))

        threshold = threshold1
        print('threshold used', threshold)

    else:
        print("Error defense")
