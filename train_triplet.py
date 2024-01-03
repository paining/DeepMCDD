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
# from models.vae_simple import VAE
from models.triplet import TripletModel

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
    parser.add_argument("--sampling", type=str, default="semihard")
    parser.add_argument("--anchor_id", type=int, nargs="*", default=0)

    return parser.parse_args()

def main():
    args = arg_parse()


    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "models"), exist_ok=True)
    set_logger("log.yaml", args.save_path)
    # logger.info(args)
    logger.info("\n".join([f"{k:15s} {v}" for k, v in vars(args).items()]))

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

    model = TripletModel(
        in_channel=1,
        hidden_channels=[32, 64, 128],
        latent_dim=16,
        in_shape=(16, 16),
        anchor_id=args.anchor_id,
        margin=1,
        sampling_type=args.sampling
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
    logger.info(
        summary(
            model,
            (1, 1, 16, 16),
            col_names=['input_size', 'output_size', 'kernel_size'],
            depth=4,
            verbose=0,
        )
    )

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
        model:TripletModel,
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
        model:TripletModel,
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

    log_loss = {
        "train": {"loss": []},
        "valid": {"loss": []},
    }

    for epoch in trange(epochs, desc="Epochs", ncols=79):
        train_loss = {"loss": []}
        model.train()
        # for x, y in tqdm(dataloader, ncols=79, desc="Train", leave=False):
        with tqdm(dataloader, desc="Train", ncols=100, leave=False) as pbar:
            for i, (x, y) in enumerate(pbar):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                loss = model.loss_function(x, y)
                if loss is not None:
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()
                else:
                    loss = 0
                if i % 20 == 0: pbar.set_postfix({"loss": loss})
                train_loss["loss"].append(loss)

        # debugging
        os.makedirs(os.path.join(log_dir, "tmp", "loss"), exist_ok=True)
        fig, ax = plt.subplots()
        for k, v in train_loss.items():
            ax.plot(v, label=k, alpha=0.7, linestyle='', marker='.')
        ax.legend()
        ax.set_title(f"Train Losses in {epoch} epoch")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(log_dir, "tmp", "loss", f"Train {epoch:03d}.png"))
        plt.close(fig)

        for k, v in train_loss.items():
            train_loss[k] = torch.mean(torch.tensor(v)).item()
        loss_str = ", ".join([f"{k}:{v:6f}" for k, v in train_loss.items()])
        with logging_redirect_tqdm():
            logger.info(f"Epoch {epoch:3d} : Train : Loss = ({loss_str})")

        valid_loss = {"loss": []}
        model.eval()

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

                for x_, y_ in zip(x_patch.split(mini_batch//4), y_patch.split(mini_batch//4)):
                    loss = model.loss_function(x_, y_)
                    valid_loss["loss"].append(loss.item() if loss is not None else 0)


        for k, v in valid_loss.items():
            valid_loss[k] = torch.mean(torch.tensor(v)).item()
        loss_str = ", ".join([f"{k}:{v:6f}" for k, v in valid_loss.items()])
        with logging_redirect_tqdm():
            logger.info(f"Epoch {epoch:3d} : Valid : Loss = ({loss_str})")

        for k, v in train_loss.items():
            writer.add_scalar(f"loss/Train:{k}", v, epoch)
        for k, v in valid_loss.items():
            writer.add_scalar(f"loss/Valid:{k}", v, epoch)

        if valid_loss["loss"] < best_loss:
            best_loss = valid_loss["loss"]
            with logging_redirect_tqdm():
                logger.info(f"best model saved at {epoch} (Loss = {best_loss:6f})")
            torch.save(model.state_dict(), os.path.join(log_dir, "models", "best.pt"))
        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, "models", f"{epoch}.pt"))
        torch.save(model.state_dict(), os.path.join(log_dir, "models", "latest.pt"))


        # draw loss curve
        for k, v in train_loss.items():
            log_loss["train"][k].append(v)
        for k, v in valid_loss.items():
            log_loss["valid"][k].append(v)

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

def test(
        model:TripletModel,
        test_loader:DataLoader,
        device:torch.device,
        log_dir:str,
        ignore_neighbor:int=0
        ):
    patch_size = 16

    model.eval()
    valid_loss = {"loss": []}
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

            loss = model.loss_function(x_patch, y_patch)

            valid_loss["loss"].append(loss)


    for k, v in valid_loss.items():
        valid_loss[k] = torch.mean(torch.tensor(v)).item()
    loss_str = ", ".join([f"{k}:{v:6f}" for k, v in valid_loss.items()])
    with logging_redirect_tqdm():
        logger.info(f"Valid : Loss = ({loss_str})")


if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()