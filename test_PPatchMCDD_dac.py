import os, argparse
import torch
import torch.nn.functional as TF
from torch.utils.data import DataLoader
import torchvision.transforms as TVT

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
import models
from resnet import wide_resnet50_2
# imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]
inv_mean = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255]
inv_std = [1 / 0.229, 1 / 0.224, 1 / 0.255]
from dataset.anomaly_dataset import AnomalyDetecionDataset, load_dataset_from_path
from einops import rearrange
import logging
from log import set_logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--gt_path", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--save_path", type=str, default="output")
    parser.add_argument("--ignore_neighbor", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--padding_mode", type=str, default="replicate")
    parser.add_argument("--gpu", type=int, default=0)

    return parser.parse_args()

def get_dataloader(data_path, gt_path):
    # set dataset transforms.
    data_transforms = TVT.Compose(
        [
            TVT.ToTensor(),
            # TVT.Normalize(mean=mean_train, std=std_train),
        ]
    )
    gt_transforms = TVT.Compose(
        [
            TVT.ToTensor(),
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

def test():

    args = arg_parse()

    os.makedirs(args.save_path, exist_ok=True)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(args.gpu)
    set_logger("log.yaml", args.save_path)
    logger.info(args)


    num_classes, num_features = 2, 1
    
    model = models.Conv_DeepMCDD_oneclass(
        num_features,
        [64, 128, 256, 10],
        #num_classes=num_classes,
        kernel_size=3,
        padding_mode="replicate"
    )
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.cuda()
    model.eval()
    conf_thr = 0.01
    print("Confidence Threshold : ", conf_thr)

    test_loader = get_dataloader(args.data_path, args.gt_path)
 
    feature_extractor = torch.nn.Sequential(
        torch.nn.Unfold(kernel_size=8, stride=8)
    )
    feature_extractor.cuda()

    GI_NP_scores = []
    BI_NP_scores = []
    BI_AP_scores = []
    GI_NP_emb = []
    BI_NP_emb = []
    BI_AP_emb = []
    Img_pred = []
    Img_label = []
    print(f"{'file_name':^20s} : {'type':^6s} : {'tp':^6s} : {'fp':^6s} : {'fn':^6s} : {'tn':^6s}")
    for x, gt, y, file_name in tqdm(test_loader, ncols=79, desc="Test"):
        with torch.no_grad():
            patchwise_gt = TF.max_pool2d(gt, kernel_size=8, stride=8)
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

            defect_mask = patchwise_gt != 0
            normal_mask = rearrange(normal_mask, "b c h w -> (b h w) c").squeeze(dim=1)
            defect_mask = rearrange(defect_mask, "b c h w -> (b h w) c").squeeze(dim=1)

            # features = feature_extractor(x.mean(dim=1, keepdim=True).cuda()) # b 64 (pH pW)

            # features = rearrange(features, "b c (h w) -> (b h w) c", h=x.shape[-2]//8)
            # dist = model(features)
            # conf, _ = torch.min(dist, dim=1)
            # scores = - dist + model.alphas
            # _, predicted = torch.max(scores, 1)

            # emb = model._forward(features)
            
            data = TF.unfold(x.mean(dim=1).cuda(), kernel_size=8, stride=8).reshape(-1, 64)
            emb = model.forward(data)
            dists = torch.mean(emb**2, dim=-1)
            predicted = torch.where(dists > model.r.item() + 0.5, 1, 0)

        # total_score = torch.cat((scores, conf.reshape(-1, 1)), dim=1)
        total_score = dists.clone()

        # positive = torch.logical_and( conf < conf_thr, predicted == 1).cpu()
        positive = (predicted == 1).cpu().detach()
        tp = torch.logical_and(defect_mask, positive).sum().item()
        fp = torch.logical_and(defect_mask == False, positive).sum().item()
        fn = torch.logical_and(defect_mask, positive == False).sum().item()
        tn = torch.logical_and(defect_mask == False, positive == False).sum().item()

        logger.info(f"{file_name[0]:20s} : {'Normal' if y[0] == 0 else 'Defect'} : {tp:6d} : {fp:6d} : {fn:6d} : {tn:6d}")

        if y == 0:
            GI_NP_scores.append(total_score[normal_mask].cpu().detach().numpy())
            GI_NP_emb.append(emb[normal_mask].cpu().detach().numpy())
        else:
            BI_NP_scores.append(total_score[normal_mask].cpu().detach().numpy())
            BI_AP_scores.append(total_score[defect_mask].cpu().detach().numpy())
            BI_NP_emb.append(emb[normal_mask].cpu().detach().numpy())
            BI_AP_emb.append(emb[defect_mask].cpu().detach().numpy())

        # Img_pred.append(torch.count_nonzero(predicted[conf < conf_thr] != 0) != 0)
        Img_pred.append(torch.count_nonzero(predicted != 0) != 0)
        Img_label.append(y)

    Img_pred = torch.tensor(Img_pred).cpu().detach().numpy()
    Img_label = torch.tensor(Img_label).cpu().detach().numpy()
    conf_mat = confusion_matrix(Img_label, Img_pred)
    disp = ConfusionMatrixDisplay(conf_mat)
    disp.plot()
    acc = accuracy_score(Img_label, Img_pred)
    plt.title(f"Image Confusion Matrix\nAccuracy={acc*100:.2f}")
    plt.savefig(os.path.join(args.save_path, "Image_Confusion_Matrix.png"))
    plt.close()

    GI_NP_scores = np.concatenate(GI_NP_scores)
    BI_NP_scores = np.concatenate(BI_NP_scores)
    BI_AP_scores = np.concatenate(BI_AP_scores)

    GI_NP_emb = np.concatenate(GI_NP_emb)
    BI_NP_emb = np.concatenate(BI_NP_emb)
    BI_AP_emb = np.concatenate(BI_AP_emb)

    gi_np_rnd = np.random.choice(len(GI_NP_emb), 2500, replace=False)
    gi_np_rnd = GI_NP_emb[gi_np_rnd]
    bi_np_rnd = np.random.choice(len(BI_NP_emb), 2500, replace=False)
    bi_np_rnd = BI_NP_emb[bi_np_rnd]
    if len(BI_AP_emb) > 5000:
        bi_ap_rnd = np.random.choice(len(BI_AP_emb), 5000, replace=False)
        bi_ap_rnd = BI_AP_emb[bi_ap_rnd]
    else:
        bi_ap_rnd = BI_AP_emb.copy()
    normals = np.concatenate([gi_np_rnd, bi_np_rnd], axis=0)
    defects = bi_ap_rnd

    fig, ax = plt.subplots(figsize=(10,8))
    ax.hist(GI_NP_scores, bins=100, alpha=0.4, label="GI_NP")
    ax.hist(BI_NP_scores, bins=100, alpha=0.4, label="BI_NP")
    ax.hist(BI_AP_scores, bins=100, alpha=0.4, label="BI_AP")
    ax.set_title(f"PPatchPosMCDD - r({model.r.item()}) distance")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(os.path.join(args.save_path, "Distance.png"))
    plt.close(fig)

    if normals.shape[-1] != 2:
        # from sklearn.manifold import Isomap as manifold_transformer
        from sklearn.manifold import TSNE as MT
        manifold_transformer = MT()
        embeddings = manifold_transformer.fit_transform(np.concatenate([normals, defects], axis=0))
        normals = embeddings[:len(normals)]
        defects = embeddings[-len(defects):]
    fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(normals[:,0], normals[:,1], alpha=0.1, c="green", marker="o", label="GI_NP")
    ax.scatter(defects[:,0], defects[:,1], alpha=0.1, c="red", marker="x", label="BI_AP")
    # colors = ["green", "red"]
    # for i, center in enumerate(model.centers):
    #     ax.scatter(center[0].item(),  center[1].item(), alpha=0.8, c=colors[i],edgecolors="black", marker="^", label=f"C{i}")
    ax.scatter(0, 0, alpha=0.8, c="green",edgecolors="black", marker="^", label=f"C0")
    ax.set_title(f"PPatchPosMCDD - Embedding({os.path.basename(args.ckpt)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, f"Embedding Space({os.path.basename(args.ckpt)})-{manifold_transformer.__class__.__name__}.png"))
    plt.close(fig)

    patches, labels, classes = np.load(
        "/home/work/.data/dac/P1_PPatch/dac_conv_train.npy", allow_pickle=True
    )
    normals = patches[labels == 0]
    normals = normals[np.random.choice(len(normals), 5000, replace=False)]
    defects = patches[labels == 1]
    normals = model._forward(torch.from_numpy(normals).cuda()).cpu().detach().numpy()
    defects = model._forward(torch.from_numpy(defects).cuda()).cpu().detach().numpy()
    if normals.shape[-1] != 2:
        # from sklearn.manifold import Isomap as manifold_transformer
        from sklearn.manifold import TSNE as MT
        manifold_transformer = MT()
        embeddings = manifold_transformer.fit_transform(np.concatenate([normals, defects], axis=0))
        normals = embeddings[:len(normals)]
        defects = embeddings[-len(defects):]

    fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(normals[:,0], normals[:,1], alpha=0.1, c="green", marker="o", label="GI_NP")
    ax.scatter(defects[:,0], defects[:,1], alpha=0.1, c="red", marker="x", label="BI_AP")
    # colors = ["green", "red"]
    # for i, center in enumerate(model.centers):
    #     ax.scatter(center[0].item(),  center[1].item(), alpha=0.8, c=colors[i],edgecolors="black", marker="^", label=f"C{i}")
    ax.scatter(0, 0, alpha=0.8, c="green",edgecolors="black", marker="^", label=f"C0")
    ax.set_title(f"PPatchPosMCDD - Embedding({os.path.basename(args.ckpt)}) - INPUT")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, f"Embedding Space({os.path.basename(args.ckpt)})-input-{manifold_transformer.__class__.__name__}.png"))
    plt.close(fig)


    for c in range(num_classes):

        fig, ax = plt.subplots()
        ax.hist(
            np.concatenate([
                GI_NP_scores[:, c][GI_NP_scores[:, -1] < conf_thr],
                BI_NP_scores[:, c][BI_NP_scores[:, -1] < conf_thr]
            ]),
            bins=100, histtype="step", alpha=0.5,
            color="green", label="Normal Patch"
        )
        ax.hist(
            BI_AP_scores[:, c][BI_AP_scores[:, -1] < conf_thr],
            bins=100, histtype="step", alpha=0.5,
            color="orange", label="Anormaly Patch"
        )
        ax.set_yscale("log")
        ax.legend()
        ax.set_title(f"Score Distribution with Class {c}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.save_path, f"Class {c} Score Distribution(conf<{conf_thr:.3f}).png"))
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.hist(
            np.concatenate([
                GI_NP_scores[:, c][np.logical_and(GI_NP_scores[:, -1] < conf_thr, GI_NP_scores[:,c] > -1)],
                BI_NP_scores[:, c][np.logical_and(BI_NP_scores[:, -1] < conf_thr, BI_NP_scores[:,c] > -1)]
            ]),
            bins=100, histtype="step", alpha=0.5,
            color="green", label="Normal Patch"
        )
        ax.hist(
            BI_AP_scores[:, c][np.logical_and(BI_AP_scores[:, -1] < conf_thr, BI_AP_scores[:,c] > -1)],
            bins=100, histtype="step", alpha=0.5,
            color="orange", label="Anormaly Patch"
        )
        ax.set_yscale("log")
        ax.legend()
        ax.set_title(f"Score Distribution with Class {c}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.save_path, f"Class {c} Score Distribution(conf<{conf_thr:.3f})_zoom.png"))
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.hist(
            GI_NP_scores[:, c], bins=100, histtype="step", alpha=0.5,
            color="green", label="GI_NP"
        )
        ax.hist(
            BI_NP_scores[:, c], bins=100, histtype="step", alpha=0.5,
            color="blue", label="BI_NP"
        )
        ax.hist(
            BI_AP_scores[:, c], bins=100, histtype="step", alpha=0.5,
            color="orange", label="BI_AP"
        )
        ax.set_yscale("log")
        ax.legend()
        ax.set_title(f"Score Distribution with Class {c}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.save_path, f"Class {c} Score Distribution.png"))
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(
        GI_NP_scores[:, -1], bins=100, histtype="step", alpha=0.5,
        color="green", label="GI_NP"
    )
    ax.hist(
        BI_NP_scores[:, -1], bins=100, histtype="step", alpha=0.5,
        color="blue", label="BI_NP"
    )
    ax.hist(
        BI_AP_scores[:, -1], bins=100, histtype="step", alpha=0.5,
        color="orange", label="BI_AP"
    )
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Confidence Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, f"Confidence Distribution.png"))
    plt.close(fig)

    scores = np.concatenate(
        [
            GI_NP_scores[:, :-1],
            BI_NP_scores[:, :-1],
            BI_AP_scores[:, :-1],
        ],
        axis=0,
    )
    conf = np.concatenate(
        [
            GI_NP_scores[:, -1],
            BI_NP_scores[:, -1],
            BI_AP_scores[:, -1],
        ],
        axis=None,
    )
    labels = np.concatenate(
        [
            0 * np.ones((len(GI_NP_scores))),
            0 * np.ones((len(BI_NP_scores))),
            1 * np.ones((len(BI_AP_scores))),
        ],
        axis=None,
    )
    pred = np.concatenate(
        [
            np.argmax(GI_NP_scores[:, :-1], axis=1),
            np.argmax(BI_NP_scores[:, :-1], axis=1),
            np.argmax(BI_AP_scores[:, :-1], axis=1),
        ]
    )
    conf_mat = confusion_matrix(labels, pred)
    disp = ConfusionMatrixDisplay(conf_mat)
    disp.plot()
    acc = accuracy_score(labels, pred)
    plt.title(f"Confusion Matrix\nAccuracy={acc*100:.2f}")
    plt.savefig(os.path.join(args.save_path, "Confusion_Matrix.png"))
    plt.close()

    x = []
    y = []
    for conf_thr in np.arange(-3, 0, 0.5):
        conf_thr = pow(10, conf_thr)
        conf_mat = confusion_matrix(labels[conf < conf_thr], pred[conf < conf_thr])
        acc = accuracy_score(labels[conf < conf_thr], pred[conf < conf_thr])
        disp = ConfusionMatrixDisplay(conf_mat)
        disp.plot()
        plt.title(f"Confusion Matrix(conf<{conf_thr:.4f})\nAccuracy={acc*100:.2f}")
        plt.savefig(os.path.join(args.save_path, f"Confusion_Matrix(conf<{conf_thr:.4f}).png"))
        plt.close()
        x.append(conf_thr)
        y.append(acc)

    fig, ax = plt.subplots()
    ax.plot(x,y)
    ax.set_xlabel("confidence_threshold")
    ax.set_xscale("log")
    ax.set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, "confidence-accuracy graph.png"))


if __name__ == "__main__":
    with logging_redirect_tqdm():
        test()