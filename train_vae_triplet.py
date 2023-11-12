import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as TF
from torch.utils.data import DataLoader
import torchvision.transforms as TVT
from dataset.anomaly_dataset import AnomalyDetecionDataset, load_dataset_from_path
from einops import rearrange

from torch.autograd import Variable
import numpy as np
import cv2
import random
import sys, os
from tqdm import tqdm, trange
from torchinfo import summary
from tqdm.contrib.logging import logging_redirect_tqdm
import matplotlib.pyplot as plt
import time

import argparse

# from vae import VAE
from models.vae_simple import VAE

from log import set_logger
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--gt_path", type=str)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="output")
    parser.add_argument("--ignore_neighbor", type=int, default=0)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--padding_mode", type=str, default="replicate")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)

    return parser.parse_args()

def main():
    args = arg_parse()


    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "models"), exist_ok=True)
    set_logger("log.yaml", args.save_path)
    logger.info(args)

    device = torch.device("cpu") if args.device is None else torch.device(args.device)

    seed = 777
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

    torch.autograd.set_detect_anomaly(True)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    model = VAE(
        in_channel=1,
        hidden_channels=[32, 64, 128],
        latent_dim=16,
        in_shape=(16, 16)
    )
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        logger.info(
            "Pretrained Model is loaded. (%s)",
            args.ckpt if len(args.ckpt) < 60 else '...' + args.ckpt[-57:]
        )
    model.to(device)
    data = torch.zeros((1,1,16,16), device=device)
    model(data)
    logger.info(model)
    logger.info(summary(model, (1, 1, 16, 16), col_names=['input_size', 'output_size', 'kernel_size']))

    if not args.eval:
        # torch.nn.utils.clip_grad.clip_grad_norm(model.parameters(), 1.)

        train_loader = get_dataloader(
            os.path.join(args.data_path, "train"),
            args.gt_path,
            batch_size=4,
            shuffle=True
            )
        test_loader = get_dataloader(os.path.join(args.data_path, "test"), args.gt_path)
        # train(
        #     model,
        #     train_loader,
        #     test_loader,
        #     args.num_epochs,
        #     args.learning_rate,
        #     device,
        #     args.save_path
        # )
        train_patch(
            model,
            train_loader,
            test_loader,
            args.num_epochs,
            args.learning_rate,
            device,
            args.save_path
        )
    else:
        test_loader = get_dataloader(os.path.join(args.data_path, "test"), args.gt_path)
        log_dir = os.path.join(
            args.save_path,
            "result",
            os.path.splitext(os.path.basename(args.ckpt))[0]
        )
        test(
            model,
            test_loader,
            device=device,
            log_dir=log_dir
        )

def get_dataloader(data_path, gt_path, batch_size=1, shuffle=False):
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
    dataset = AnomalyDetecionDataset(
        data_path,
        *load_dataset_from_path(data_path, [["good"], []], gt_path, True),
        transform=data_transforms,
        gt_transform=gt_transforms,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
    )
    return dataloader

### https://github.com/haofuml/cyclical_annealing
def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L    


def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 1.0/(1.0+ np.exp(- (v*12.-6.)))
            v += step
            i += 1
    return L    


#  function  = 1 − cos(a), where a scans from 0 to pi/2
import math
def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    # transform into [0, pi] for plots: 

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 0.5-.5*math.cos(v*math.pi)
            v += step
            i += 1
    return L    

def train(
        model:VAE,
        train_loader:DataLoader,
        test_loader:DataLoader,
        epochs:int,
        learing_rate:float,
        device:torch.device,
        log_dir:str
        ):
    os.makedirs(os.path.join(log_dir, "tensorboard"), exist_ok=True)
    writer = SummaryWriter(os.path.join(log_dir, "tensorboard"))
    optimizer = optim.Adam(model.parameters(), lr=learing_rate)
    patch_size = 16
    stride = 8
    best_loss = 1e10
    mini_batch = 1024
    logger.info(f"Mini-Batch Size : {mini_batch}")

    # beta_arr = np.concatenate([
    #     np.zeros((epochs//4,)),
    #     0.001 * np.ones((epochs//4,)),
    #     np.linspace(0.001, 1, epochs//4, endpoint=True),
    #     np.ones((epochs - (3*(epochs//4)),))
    # ])

    beta_arr = frange_cycle_sigmoid(0, 1, epochs, 4, 0.5)
    fig, ax = plt.subplots()
    ax.plot(beta_arr, label="beta")
    ax.set_title("Cycling Beta Annealing")
    ax.set_ylabel("beta")
    ax.set_xlabel("epoch")
    ax.grid(True, "both", "both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "beta.png"))

    for epoch in trange(epochs, desc="Epochs", ncols=79):
        # beta = min(1, max(0.001, 20.*(epoch-10)/epochs))
        beta:float = beta_arr[epoch].item()
        train_loss = []
        train_re = []
        train_kld = []
        max_grad = 0
        model.train()
        for x, gt, y, filename in tqdm(train_loader, ncols=79, desc="Train", leave=False):
            with torch.no_grad():
                x = x.to(device)
                h1 = int((x.shape[-2] - patch_size)/stride + 1)
                w1 = int((x.shape[-1] - patch_size)/stride + 1)
                x_patch = TF.unfold(x.mean(dim=1, keepdim=True), kernel_size=patch_size, stride=patch_size//2)
                x_patch = rearrange(
                    x_patch,
                    "b (h2 w2) (h1 w1) -> (b h1 w1) 1 h2 w2",
                    h2=patch_size,
                    w2=patch_size,
                    h1=h1,
                    w1=w1
                )

            idx =  torch.randperm(x_patch.shape[0])
            for batch in torch.split(x_patch[idx], mini_batch):
                optimizer.zero_grad()
                x_, mu, logvar = model(batch)

                re = model.RE(x_, batch)
                kld = model.KLD(mu, logvar)
                loss = re + beta * kld
                train_loss.append(loss.item())
                train_re.append(re.item())
                train_kld.append(kld.item())

                loss.backward()
                for param in model.parameters():
                    grad_ = torch.max(param.grad).item()
                    if grad_ > max_grad:
                        max_grad = grad_
                optimizer.step()

            # optimizer.zero_grad()
            # x_, mu, logvar = model(x_patch)

            # re = model.RE(x_, x_patch)
            # kld = beta * model.KLD(mu, logvar)
            # loss = re + kld

            # train_loss.append(loss.item())
            # train_re.append(re.item())
            # train_kld.append(kld.item())

            # loss.backward()
            # for param in model.parameters():
            #     grad_ = torch.max(param.grad).item()
            #     if grad_ > grad_max:
            #         grad_max = grad_
            # optimizer.step()

        train_loss = torch.mean(torch.tensor(train_loss)).item()
        train_re = torch.mean(torch.tensor(train_re)).item()
        train_kld = torch.mean(torch.tensor(train_kld)).item()
        logger.info(f"Epoch {epoch:3d} : Train : Loss = {train_loss:10f}(RE:{train_re:10f}, KLD:{train_kld:10f})")

        valid_loss = []
        valid_re = []
        valid_kld = []
        model.eval()
        
        if epoch % 5 == 0:
            visualize(model, test_loader, device, os.path.join(log_dir, "tmp", f"{epoch:03d}"))
        for x, gt, y, filename in tqdm(test_loader, ncols=79, desc="Test", leave=False):
            with torch.no_grad():
                x = x.to(device)
                h1 = int((x.shape[-2] - patch_size)/stride + 1)
                w1 = int((x.shape[-1] - patch_size)/stride + 1)
                x_patch = TF.unfold(x.mean(dim=1, keepdim=True), kernel_size=patch_size, stride=patch_size//2)
                x_patch = rearrange(
                    x_patch,
                    "b (h2 w2) (h1 w1) -> (b h1 w1) 1 h2 w2",
                    h2=patch_size,
                    w2=patch_size,
                    h1=h1,
                    w1=w1
                )

                x_, mu, logvar = model(x_patch)

                re = model.RE(x_, x_patch)
                kld = beta * model.KLD(mu, logvar)
                loss = re + kld
                valid_loss.append(loss.item())
                valid_re.append(re.item())
                valid_kld.append(kld.item())

        valid_loss = torch.mean(torch.tensor(valid_loss)).item()
        valid_re = torch.mean(torch.tensor(valid_re)).item()
        valid_kld = torch.mean(torch.tensor(valid_kld)).item()
        logger.info(f"Epoch {epoch:3d} : Valid : Loss = {valid_loss:10f}(RE:{valid_re:10f}, KLD:{valid_kld:10f})")

        writer.add_scalar("loss/Train:RE", train_re, epoch)
        writer.add_scalar("loss/Train:KLD", train_kld, epoch)
        writer.add_scalar("loss/Train", train_loss, epoch)
        writer.add_scalar("loss/Valid:RE", valid_re, epoch)
        writer.add_scalar("loss/Valid:KLD", valid_kld, epoch)
        writer.add_scalar("loss/Valid", valid_loss, epoch)
        writer.add_scalar("Param/beta", beta, epoch)
        writer.add_scalar("Param/max_grad", max_grad, epoch)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(log_dir, "models", "best.pt"))
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, "models", f"{epoch}.pt"))
        torch.save(model.state_dict(), os.path.join(log_dir, "models", "latest.pt"))


def train_patch(
        model:VAE,
        train_loader:DataLoader,
        test_loader:DataLoader,
        epochs:int,
        learing_rate:float,
        device:torch.device,
        log_dir:str,
        ignore_neighbor:int=0
        ):
    os.makedirs(os.path.join(log_dir, "tensorboard"), exist_ok=True)
    writer = SummaryWriter(os.path.join(log_dir, "tensorboard"))
    optimizer = optim.Adam(model.parameters(), lr=learing_rate)
    patch_size = 16
    stride = 8
    best_loss = 1e10
    mini_batch = 1024
    logger.info(f"Mini-Batch Size : {mini_batch}")

    # Make Patch-dataset
    patch_dataset = []
    label_dataset = []
    for x, gt, y, filename in tqdm(train_loader, ncols=79, desc="Extracting", leave=False):
        with torch.no_grad():
            h1 = int((x.shape[-2] - patch_size)/stride + 1)
            w1 = int((x.shape[-1] - patch_size)/stride + 1)
            x_patch = TF.unfold(x.mean(dim=1, keepdim=True), kernel_size=patch_size, stride=stride)
            x_patch = rearrange(
                x_patch,
                "b (h2 w2) (h1 w1) -> (b h1 w1) 1 h2 w2",
                h2=patch_size,
                w2=patch_size,
                h1=h1,
                w1=w1
            )
            y_patch = TF.unfold(gt, kernel_size=patch_size, stride=stride).sum(dim=1)
            y_patch = rearrange(y_patch, "b (h1 w1) -> (b h1 w1)", h1=h1, w1=w1)
            idx = torch.logical_or(y_patch == 0, y_patch >= 7)
            y_patch = y_patch != 0
            patch_dataset.append(x_patch[idx])
            label_dataset.append(y_patch[idx])
    patch_dataset = torch.cat(patch_dataset, dim=0)
    label_dataset = torch.cat(label_dataset, dim=0)
    patch_dataset = TensorDataset(patch_dataset, label_dataset)
    dataloader = DataLoader(patch_dataset, mini_batch, shuffle=True, num_workers=4, drop_last=True)

    # beta_arr = frange_cycle_sigmoid(0, 1, epochs*len(dataloader), 4, 0.9)
    # beta_arr = 1e-4 * np.ones((epochs*len(dataloader),), dtype=np.float32)
    gamma_arr = frange_cycle_sigmoid(0, 1, epochs*len(dataloader), 4, 0.9)

    fig, ax = plt.subplots()
    # ax.plot(np.linspace(0, epochs, len(beta_arr)), beta_arr, label="beta", color="orange")
    ax2 = ax.twinx()
    ax2.plot(np.linspace(0, epochs, len(gamma_arr)), gamma_arr, label="gamma", color="blue")
    # ax.set_title("Cycling Beta Annealing")
    # ax.set_ylabel("beta")
    lines = [line for line in ax.get_lines()] + [line for line in ax2.get_lines()]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels)
    ax.set_xlabel("epoch")
    ax.grid(True, "both", "both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "beta.png"))
    plt.close(fig)

    log_loss = {
        "train": {"loss": [], "re": [], "kld": [], "triplet": []},
        "valid": {"loss": [], "re": [], "kld": [], "triplet": []},
    }

    for epoch in trange(epochs, desc="Epochs", ncols=79):
        # beta = min(1, max(0.001, 20.*(epoch-10)/epochs))
        iter = epoch * len(dataloader)
        train_loss = []
        train_re = []
        # train_kld = []
        train_triplet = []
        max_grad = 0
        model.train()
        # for x, y in tqdm(dataloader, ncols=79, desc="Train", leave=False):
        with tqdm(dataloader, desc="Train", ncols=100, leave=False) as pbar:
            for x, y in pbar:
                # beta:float = beta_arr[iter].item()
                gamma:float = gamma_arr[iter].item()
                iter += 1
                # loss, re, kld, triplet = train_iter(model, x, y, beta, gamma, optimizer, device)
                loss, re, triplet = train_iter2(model, x, y, gamma, optimizer, device)
                if iter % 20 == 0:
                    pbar.set_postfix({"loss": loss, "re": re, "triplet": triplet})

                train_loss.append(loss)
                train_re.append(re)
                # train_kld.append(kld)
                train_triplet.append(triplet)

        # debugging
        os.makedirs(os.path.join(log_dir, "tmp", "loss"), exist_ok=True)
        fig, ax = plt.subplots()
        ax.plot(train_loss, label="loss", alpha=0.7)
        ax.plot(train_re, label="re", alpha=0.7)
        # ax.plot(train_kld, label="kld", alpha=0.7)
        ax.plot(train_triplet, label="triplet", alpha=0.7)
        ax.legend()
        ax.set_title(f"Train Losses in {epoch} epoch")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(log_dir, "tmp", "loss", f"Train {epoch:03d}.png"))
        plt.close(fig)

        train_loss = torch.mean(torch.tensor(train_loss)).item()
        train_re = torch.mean(torch.tensor(train_re)).item()
        # train_kld = torch.mean(torch.tensor(train_kld)).item()
        train_triplet = torch.mean(torch.tensor(train_triplet)).item()
        with logging_redirect_tqdm():
            logger.info(
                f"Epoch {epoch:3d} : Train : Loss = {train_loss:4f}("
                f"RE:{train_re:4f}, "
                # f"KLD:{train_kld:4f}, "
                f"Triplet:{train_triplet:4f}"
                ")")

        valid_loss = []
        valid_re = []
        # valid_kld = []
        valid_triplet = []
        model.eval()
        
        if epoch % 5 == 0:
            visualize(model, test_loader, device, os.path.join(log_dir, "tmp", f"{epoch:03d}"))
        for x, gt, y, filename in tqdm(test_loader, ncols=79, desc="Test", leave=False):
            with torch.no_grad():
                x = x.to(device)
                h1 = int((x.shape[-2] - patch_size)/stride + 1)
                w1 = int((x.shape[-1] - patch_size)/stride + 1)
                x_patch = TF.unfold(x.mean(dim=1, keepdim=True), kernel_size=patch_size, stride=patch_size//2)
                x_patch = rearrange(
                    x_patch,
                    "b (h2 w2) (h1 w1) -> (b h1 w1) 1 h2 w2",
                    h2=patch_size,
                    w2=patch_size,
                    h1=h1,
                    w1=w1
                )
                y_patch = TF.unfold(gt, kernel_size=patch_size, stride=stride).sum(dim=1)
                y_patch = rearrange(y_patch, "b (h1 w1) -> (b h1 w1)", h1=h1, w1=w1)
                if ignore_neighbor != 0:
                    expand = rearrange(y_patch, "(b h1 w1) -> b 1 h1 w1", h1=h1, w1=w1)
                    expand = TF.max_pool2d(
                        expand,
                        kernel_size=2*ignore_neighbor+1,
                        padding=ignore_neighbor
                    )
                    expand = rearrange(expand, "b 1 h w -> (b h w)")
                else:
                    expand = y_patch.clone()
                idx = torch.logical_or(expand == 0, y_patch >= 7)
                y_patch = y_patch != 0

                x_patch = x_patch[idx]
                y_patch = y_patch[idx]

                # loss, re, kld, triplet = train_iter(model, x_patch, y_patch, beta, gamma, None, device)
                loss, re, triplet = train_iter2(model, x_patch, y_patch, gamma, None, device)

                valid_loss.append(loss)
                valid_re.append(re)
                # valid_kld.append(kld)
                valid_triplet.append(triplet)

        valid_loss = torch.mean(torch.tensor(valid_loss)).item()
        valid_re = torch.mean(torch.tensor(valid_re)).item()
        # valid_kld = torch.mean(torch.tensor(valid_kld)).item()
        valid_triplet = torch.mean(torch.tensor(valid_triplet)).item()
        with logging_redirect_tqdm():
            logger.info(
                f"Epoch {epoch:3d} : Valid : Loss = {valid_loss:4f}("
                f"RE:{valid_re:4f}, "
                # f"KLD:{valid_kld:4f}, "
                f"Triplet:{valid_triplet:4f}"
                ")")

        writer.add_scalar("loss/Train:RE", train_re, epoch)
        # writer.add_scalar("loss/Train:KLD", train_kld, epoch)
        # writer.add_scalar("loss/Train:beta*KLD", beta*train_kld, epoch)
        writer.add_scalar("loss/Train:Triplet", train_triplet, epoch)
        writer.add_scalar("loss/Train", train_loss, epoch)
        writer.add_scalar("loss/Valid:RE", valid_re, epoch)
        # writer.add_scalar("loss/Valid:KLD", valid_kld, epoch)
        # writer.add_scalar("loss/Valid:beta*KLD", beta * valid_kld, epoch)
        writer.add_scalar("loss/Valid:Triplet", valid_triplet, epoch)
        writer.add_scalar("loss/Valid", valid_loss, epoch)
        # writer.add_scalar("Param/beta", beta, epoch)
        writer.add_scalar("Param/gamma", gamma, epoch)
        # writer.add_scalar("Param/max_grad", max_grad, epoch)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(log_dir, "models", "best.pt"))
        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, "models", f"{epoch}.pt"))
        torch.save(model.state_dict(), os.path.join(log_dir, "models", "latest.pt"))


        # draw loss curve
        log_loss["train"]["loss"].append(train_loss)
        log_loss["train"]["re"].append(train_re)
        # log_loss["train"]["kld"].append(train_kld)
        log_loss["train"]["triplet"].append(train_triplet)
        log_loss["valid"]["loss"].append(valid_loss)
        log_loss["valid"]["re"].append(valid_re)
        # log_loss["valid"]["kld"].append(valid_kld)
        log_loss["valid"]["triplet"].append(valid_triplet)

        fig, ax = plt.subplots()
        for k, v in log_loss["train"].items():
            ax.plot(v, label=f"train:{k}", alpha=0.6, linestyle="-")
        for k, v in log_loss["valid"].items():
            ax.plot(v, label=f"valid:{k}", alpha=0.6, linestyle="--")
        ax.legend()
        ax.set_title("loss curve")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, "both", "both", alpha=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(log_dir, "loss curve.png"))
        plt.close(fig)

from pytorch_metric_learning import miners, losses
def make_triplet_all(x, y, anchor_id):
    anchor = x[y == anchor_id]
    positive = x[y == anchor_id]
    negative = x[y != anchor_id]

    if len(anchor) == 0 or len(negative) == 0:
        return None

    n_a = anchor.shape[0]
    n_p = positive.shape[0]
    n_n = negative.shape[0]
    out_ch = x.shape[-1]
    anchor = anchor.reshape(n_a, 1, 1, -1).repeat(1, n_p, n_n, 1).reshape(-1, out_ch)
    positive = positive.reshape(1, n_p, 1, -1).repeat(n_a, 1, n_n, 1).reshape(-1, out_ch)
    negative = negative.reshape(1, 1, n_n, -1).repeat(n_a, n_p, 1, 1).reshape(-1,out_ch)
    return anchor, positive, negative

def make_triplet_softhard(x, y, anchor_id, margin):
    anchor = x[y == anchor_id]
    positive = x[y == anchor_id]
    negative = x[y != anchor_id]

    if len(anchor) == 0 or len(negative) == 0:
        return None

    dist_pos = torch.cdist(anchor, positive)
    dist_neg = torch.cdist(anchor, negative)
    # Find Easy Positive
    min_dist_pos, min_idx_pos = torch.kthvalue(dist_pos, 2, dim=1)
    # Find Semi-Hard Negative
    min_dist_pos = min_dist_pos.unsqueeze(dim=1)
    anchor_idx, negative_idx = torch.where(
        torch.logical_and(
            dist_neg < min_dist_pos + margin,
            dist_neg >= min_dist_pos
        )
    )
    if len(anchor_idx) == 0 or len(negative_idx) == 0:
        return None
    positive_idx = min_idx_pos[anchor_idx]

    return anchor[anchor_idx], positive[positive_idx], negative[negative_idx]

def train_iter(model, x, y, beta, gamma, optimizer, device):
    x = x.to(device)
    if optimizer: optimizer.zero_grad()
    x_, mu, logvar = model(x)

    triplet = make_triplet_all(mu, y, 0)
    if triplet is not None:
        anchor, positive, negative = triplet
        triplet_loss = gamma * TF.triplet_margin_loss(anchor, positive, negative)
    else:
        triplet_loss = torch.tensor(0, device=device)

    re = model.RE(x_, x)
    kld = beta * model.KLD(mu, logvar)
    loss = re + kld + triplet_loss

    if optimizer: 
        loss.backward()
        optimizer.step()
    return loss.item(), re.item(), kld.item(), triplet_loss.item()

def train_iter2(model, x, y, gamma, optimizer, device):
    x = x.to(device)
    if optimizer: optimizer.zero_grad()
    x_, mu, logvar = model(x)

    # triplet = make_triplet_all(mu, y, 0)
    triplet = make_triplet_softhard(mu, y, 0, 1)
    if triplet is not None:
        anchor, positive, negative = triplet
        triplet_loss = TF.triplet_margin_loss(anchor, positive, negative)
    else:
        triplet_loss = torch.tensor(0, device=device)

    re = model.RE(x_, x)
    loss = re + gamma * triplet_loss

    if optimizer: 
        loss.backward()
        optimizer.step()
    return loss.item(), re.item(), triplet_loss.item()

def visualize(model, test_loader, device, log_dir):
    patch_size = 16
    os.makedirs(log_dir, exist_ok=True)
    for x, gt, y, filename in tqdm(test_loader, ncols=79, desc="Test", leave=False):
        with torch.no_grad():
            x = x.to(device)
            h1 = int((x.shape[-2] - patch_size)/patch_size + 1)
            w1 = int((x.shape[-1] - patch_size)/patch_size + 1)
            x_patch = TF.unfold(
                x.mean(dim=1, keepdim=True),
                kernel_size=patch_size,
                stride=patch_size
            )
            x_patch = rearrange(
                x_patch,
                "b (h2 w2) (h1 w1) -> (b h1 w1) 1 h2 w2",
                h2=patch_size,
                w2=patch_size,
                h1=h1,
                w1=w1
            )

            x_, mu, logvar = model(x_patch)

            x__ = rearrange(
                x_,
                "(b h1 w1) 1 h2 w2 -> b 1 (h1 h2) (w1 w2)",
                h1=h1, h2=patch_size, w1=w1, w2=patch_size
            )
            x__ = x__.detach().cpu().numpy()
            x__ = np.transpose(x__[0], (1, 2, 0))
            x = np.transpose(x[0].detach().cpu().numpy(), (1, 2, 0))
            cv2.imwrite(os.path.join(log_dir, f"{filename[0]}_RE.png"), x__*255)
            cv2.imwrite(os.path.join(log_dir, f"{filename[0]}.png"), x*255)

            
def test(
        model:VAE,
        test_loader:DataLoader,
        device:torch.device,
        log_dir:str,
        ignore_neighbor:int=0
        ):
    patch_size = 16

    model.eval()
    valid_re = []
    valid_kld = []
    valid_triplet = []
    visualize(model, test_loader, device, os.path.join(log_dir, "tmp"))
    for x, gt, y, filename in tqdm(test_loader, ncols=79, desc="Test", leave=False):
        with torch.no_grad():
            x = x.to(device)
            h1 = int((x.shape[-2] - patch_size)/patch_size + 1)
            w1 = int((x.shape[-1] - patch_size)/patch_size + 1)
            x_patch = TF.unfold(
                x.mean(dim=1, keepdim=True),
                kernel_size=patch_size,
                stride=patch_size
            )
            x_patch = rearrange(
                x_patch,
                "b (h2 w2) (h1 w1) -> (b h1 w1) 1 h2 w2",
                h2=patch_size,
                w2=patch_size,
                h1=h1,
                w1=w1
            )
            y_patch = TF.unfold(gt, kernel_size=patch_size, stride=patch_size).sum(dim=1)
            y_patch = rearrange(y_patch, "b (h1 w1) -> (b h1 w1)", h1=h1, w1=w1)
            
            if ignore_neighbor != 0:
                expand = rearrange(y_patch, "(b h1 w1) -> b 1 h1 w1", h1=h1, w1=w1)
                expand = TF.max_pool2d(
                    expand,
                    kernel_size=2*ignore_neighbor+1,
                    padding=ignore_neighbor
                )
                expand = rearrange(expand, "b 1 h w -> (b h w)")
            else:
                expand = y_patch.clone()
            idx = torch.logical_or(expand == 0, y_patch >= 7)
            y_patch = y_patch != 0

            x_patch = x_patch[idx]
            y_patch = y_patch[idx]


            # x_, mu, logvar = model(x_patch)

            # re = model.RE(x_, x_patch)
            # kld = model.KLD(mu, logvar)

            loss, re, triplet = train_iter2(model, x_patch, y_patch, 1, None, device)
            valid_re.append(re.item())
            # valid_kld.append(kld.item())
            valid_triplet.append(triplet.item())

    valid_re = torch.mean(torch.tensor(valid_re)).item()
    # valid_kld = torch.mean(torch.tensor(valid_kld)).item()
    valid_triplet = torch.mean(torch.tensor(valid_triplet)).item()
    logger.info(f"Valid : RE:{valid_re:6f}, KLD:{valid_kld:6f}, TRIPLET:{valid_triplet:6f}")

if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()