import argparse
import os
from dataset.anomaly_dataset import (
    AnomalyDetecionDataset,
    load_dataset_from_path,
)
import torch
import torch.nn.functional as TF
from torch.utils.data import DataLoader
from torchvision import transforms 
from tqdm import tqdm

from resnet import wide_resnet50_2

import numpy as np

from einops import rearrange
import logging
logger = logging.getLogger(__name__)

# imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]
inv_mean = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255]
inv_std = [1 / 0.229, 1 / 0.224, 1 / 0.255]


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--gt_path", type=str)
    parser.add_argument("--ignore_neighbor", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--padding_mode", type=str, default="replicate")

    return parser.parse_args()

def get_dataloader(data_path, gt_path):
    # set dataset transforms.
    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train, std=std_train),
        ]
    )
    gt_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train_dataset = AnomalyDetecionDataset(
        data_path,
        *load_dataset_from_path(data_path, [["good"], []], gt_path, True),
        transform=data_transforms,
        gt_transform=gt_transforms,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=4
    )
    return train_loader

if __name__ == "__main__":
    args = arg_parse()

    if torch.cuda.is_available() and args.device < torch.cuda.device_count():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    train_loader = get_dataloader(args.data_path, args.gt_path)

    feature_extractor = wide_resnet50_2(pretrained=True, padding_mode=args.padding_mode)
    feature_extractor.to(device)

    GI_NP = []
    BI_NP = []
    BI_AP = []
    
    GI_NP_masks = []
    BI_NP_masks = []
    BI_AP_masks = []
    for x, gt, y, file_name in tqdm(train_loader, ncols=79, desc="Sampling"):
        # patchwise_gt = TF.max_pool2d(gt, kernel_size=8, stride=8)
        patchwise_gt = (
            TF.unfold(gt, kernel_size=8, stride=8)
            .sum(dim=-2)
            .reshape(gt.shape[0],-1, gt.shape[-2]//8, gt.shape[-1]//8)
            )
        if args.ignore_neighbor != 0:
            kernel_size = 2*args.ignore_neighbor + 1
            normal_mask = (
                TF.max_pool2d(
                    patchwise_gt,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=args.ignore_neighbor
                ) == 0
            )
        else:
            normal_mask = patchwise_gt == 0

        defect_mask = patchwise_gt >= 7
        normal_mask = rearrange(normal_mask, "b c h w -> (b h w) c").squeeze(dim=1)
        defect_mask = rearrange(defect_mask, "b c h w -> (b h w) c").squeeze(dim=1)
        if y[0] == 0:
            GI_NP_masks.append(normal_mask.detach().cpu().numpy())
            BI_NP_masks.append(np.zeros(normal_mask.shape, dtype=np.bool_))
            BI_AP_masks.append(np.zeros(normal_mask.shape, dtype=np.bool_))
        else:
            GI_NP_masks.append(np.zeros(normal_mask.shape, dtype=np.bool_))
            BI_NP_masks.append(normal_mask.detach().cpu().numpy())
            BI_AP_masks.append(defect_mask.detach().cpu().numpy())
    
    nPatches = [ m.size for m in GI_NP_masks]
    masks = np.concatenate(GI_NP_masks, axis=None)
    idx = np.random.choice(np.where(masks != 0)[0], 250000, replace=False)
    masks = np.zeros_like(masks)
    masks[idx] = True
    GI_NP_masks = []
    for n in nPatches:
        m, masks = np.split(masks, [n])
        GI_NP_masks.append(m)

    nPatches = [ m.size for m in BI_NP_masks]
    masks = np.concatenate(BI_NP_masks, axis=None)
    idx = np.random.choice(np.where(masks != 0)[0], 250000, replace=False)
    masks = np.zeros_like(masks)
    masks[idx] = True
    BI_NP_masks = []
    for n in nPatches:
        m, masks = np.split(masks, [n])
        BI_NP_masks.append(m)


    for i, (x, gt, y, file_name),  in enumerate(tqdm(train_loader, ncols=79, desc="Featuring")):
        x = x.to(device)
        with torch.no_grad():
            embedding = None
            features = feature_extractor(x)
            for l, feature in enumerate(features):
                if l == 0: continue
                if args.padding_mode == "zeros":
                    o = TF.avg_pool2d(TF.pad(feature, (1,1,1,1), "constant"), 3, 1, 0)
                elif args.padding_mode == "replicate":
                    o = TF.avg_pool2d(TF.pad(feature, (1,1,1,1), args.padding_mode), 3, 1, 0)
                else:
                    logger.error( f"Unknown padding mode : {args.padding_mode}")
                    o = feature
                embedding = (
                    o if embedding is None 
                    else torch.cat(
                        (embedding, TF.interpolate(o, embedding.shape[2:], mode='bilinear')),
                        dim=1
                    )
                )

        embedding = rearrange(embedding, "b c h w -> (b h w) c")

        # normal_mask = normal_mask_list[i]
        # defect_mask = defect_mask_list[i]
        
        # if y == 0:
        #     GI_NP.append(embedding[normal_mask].detach().cpu().numpy())
        # else:
        #     BI_NP.append(embedding[normal_mask].detach().cpu().numpy())
        #     BI_AP.append(embedding[defect_mask].detach().cpu().numpy())

        GI_NP.append(embedding[GI_NP_masks[i]].detach().cpu().numpy())
        BI_NP.append(embedding[BI_NP_masks[i]].detach().cpu().numpy())
        BI_AP.append(embedding[BI_AP_masks[i]].detach().cpu().numpy())

    GI_NP = np.concatenate(GI_NP, axis=0)
    BI_NP = np.concatenate(BI_NP, axis=0)
    BI_AP = np.concatenate(BI_AP, axis=0)

    total = np.concatenate([GI_NP, BI_NP, BI_AP], axis=0)
    mean, std = np.mean(total, axis=0), np.std(total, axis=0)
    GI_NP = (GI_NP - mean) / std
    BI_NP = (BI_NP - mean) / std
    BI_AP = (BI_AP - mean) / std

    np.save("dac_1class.npy", (BI_AP, np.zeros((len(BI_AP),)), 1), allow_pickle=True)

    GI_NP_labels = 0 * np.ones((len(GI_NP),))
    BI_NP_labels = 0 * np.ones((len(BI_NP),))
    BI_AP_labels = 1 * np.ones((len(BI_AP),))

    features = np.concatenate((GI_NP, BI_NP, BI_AP), axis=0)
    labels = np.concatenate((GI_NP_labels, BI_NP_labels, BI_AP_labels), axis=0)
    np.save("dac_2class.npy", (features, labels, 2), allow_pickle=True)

    GI_NP_labels = 0 * np.ones((len(GI_NP),))
    BI_NP_labels = 1 * np.ones((len(BI_NP),))
    BI_AP_labels = 2 * np.ones((len(BI_AP),))

    features = np.concatenate((GI_NP, BI_NP, BI_AP), axis=0)
    labels = np.concatenate((GI_NP_labels, BI_NP_labels, BI_AP_labels), axis=0)
    np.save("dac_3class.npy", (features, labels, 3), allow_pickle=True)