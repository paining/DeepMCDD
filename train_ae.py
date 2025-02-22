import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard.writer import SummaryWriter
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
# from vae_simple import VAE
# from ae_simple import AE
from models.ae_simple import AE

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

    model = AE(
        in_channel=1,
        hidden_channels=[32, 64, 128],
        latent_dim=128,
        in_shape=(16, 16)
    )
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.to(device)
    data = torch.zeros((1,1,16,16), device=device)
    model(data)
    logger.info(model)
    logger.info(summary(model, (1, 1, 16, 16)))

    if not args.eval:
        # torch.nn.utils.clip_grad.clip_grad_norm(model.parameters(), 0.1)

        train_loader = get_dataloader(
            os.path.join(args.data_path, "train"),
            args.gt_path,
            batch_size=4,
            shuffle=True
            )
        test_loader = get_dataloader(os.path.join(args.data_path, "test"), args.gt_path)
        train(
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

def train(
        model,
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
    stride = patch_size//2
    best_loss = 1e10
    mini_batch = 2048

    for epoch in trange(epochs, desc="Epochs", ncols=79):
        train_loss = []
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

            # idx =  torch.randperm(x_patch.shape[0])
            # for batch in torch.split(x_patch[idx], mini_batch):
            #     optimizer.zero_grad()
            #     x_, mu, logvar = model(batch)

            #     re = model.RE(x_, batch)
            #     kld = beta * model.KLD(mu, logvar)
            #     loss = re + kld
            #     train_loss.append(loss.item())
            #     train_re.append(re.item())
            #     train_kld.append(kld.item())

            #     loss.backward()
            #     optimizer.step()

            optimizer.zero_grad()
            x_ = model(x_patch)

            loss = model.RE(x_, x_patch)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        train_loss = torch.mean(torch.tensor(train_loss)).item()
        logger.info(f"Epoch {epoch:3d} : Train : Loss = {train_loss:10f}")

        valid_loss = []
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

                x_ = model(x_patch)

                loss = model.RE(x_, x_patch)
                valid_loss.append(loss.item())

        valid_loss = torch.mean(torch.tensor(valid_loss)).item()
        logger.info(f"Epoch {epoch:3d} : Valid : Loss = {valid_loss:10f}")

        writer.add_scalar("loss/Train", train_loss, epoch)
        writer.add_scalar("loss/Valid", valid_loss, epoch)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(log_dir, "models", "best.pt"))
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, "models", f"{epoch}.pt"))

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

            x_ = model(x_patch)

            x__ = rearrange(
                x_,
                "(b h1 w1) 1 h2 w2 -> b 1 (h1 h2) (w1 w2)",
                h1=h1, h2=patch_size, w1=w1, w2=patch_size
            )
            x = rearrange(
                x_patch,
                "(b h1 w1) 1 h2 w2 -> b 1 (h1 h2) (w1 w2)",
                h1=h1, h2=patch_size, w1=w1, w2=patch_size
            )
            x__ = x__.detach().cpu().numpy()
            x__ = np.transpose(x__[0], (1, 2, 0))
            x = np.transpose(x[0].detach().cpu().numpy(), (1, 2, 0))
            cv2.imwrite(os.path.join(log_dir, f"{filename[0]}_RE.png"), x__*255)
            cv2.imwrite(os.path.join(log_dir, f"{filename[0]}.png"), x*255)

            
def test(
        model,
        test_loader:DataLoader,
        device:torch.device,
        log_dir:str
        ):
    patch_size = 16

    model.eval()
    valid_re = []
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

            x_ = model(x_patch)

            re = model.RE(x_, x_patch)
            valid_re.append(re.item())

    valid_re = torch.mean(torch.tensor(valid_re)).item()
    logger.info(f"Valid : RE:{valid_re:10f}")

if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()