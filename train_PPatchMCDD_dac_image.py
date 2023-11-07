import os, argparse, datetime
import torch
import torch.nn.functional as TF
from torch.utils.data import DataLoader
import torchvision.transforms as TVT
from torch.utils.tensorboard.writer import SummaryWriter
from torchinfo import summary
import numpy as np
from dataset.anomaly_dataset import AnomalyDetecionDataset, load_dataset_from_path
from einops import rearrange

from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
from sklearn.metrics import balanced_accuracy_score

import models
from dataloader_table import get_table_data
from utils import compute_confscores, compute_metrics, print_ood_results, print_ood_results_total

import logging
from log import set_logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./table_data/', help='path to dataset')
parser.add_argument('--test_path', default=None, help='path to test dataset')
parser.add_argument('--outdir', default='./output/', help='folder to output results')
parser.add_argument('--batch_size', type=int, default=200, help='batch size for data loader')
parser.add_argument('--num_epochs', type=int, default=10, help='the number of epochs for training sc-layers')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate of Adam optimizer')
parser.add_argument('--alpha', type=float, default=0.5, help='boundary margin')
parser.add_argument('--reg_lambda', type=float, default=1.0, help='regularization coefficient')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')

args = parser.parse_args()

set_logger("log.yaml", args.outdir)
logger.info(args)

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
    train_dataset = AnomalyDetecionDataset(
        data_path,
        *load_dataset_from_path(data_path, [["good"], []], gt_path, True),
        transform=data_transforms,
        gt_transform=gt_transforms,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
    )
    return train_loader

def main():
    outdir = os.path.join(args.outdir, args.net_type + '_' + args.dataset)

    if os.path.isdir(outdir) == False:
        os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(args.gpu)

    train_loader = get_dataloader(args.data_path, args.gt_path, args.batch_size, True)
    if args.test_path is None:
        args.test_path = args.data_path
    test_loader = get_dataloader(args.test_path, args.gt_path)

    time_str = datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')
    logdir = f"runs/{time_str}_{os.path.basename(args.outdir)}"
    writer = SummaryWriter(log_dir=logdir)
    max_acc = 0
    min_loss = 1e10

    patch_size = 8
    model = models.Conv_DeepMCDD_oneclass(
        1,
        [64, 128, 256, 10],
        kernel_size=3,
        padding_mode="replicate")
    model.cuda()

    logger.info(summary(model, (1, 64)))

    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)

    pixel_patch_featuring = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    patch_gt_pooling = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    ignore_neighbor = torch.nn.MaxPool2d(kernel_size=5, padding=2)

    idacc_list, oodacc_list = [], []
    for epoch in trange(args.num_epochs, desc="epochs", leave=False):
        model.train()
        total_loss = 0.0
        total_push_loss = 0.0
        total_pull_loss = 0.0
        total_step = 0

        for i, (data, gt, y, filename) in enumerate(tqdm(train_loader, desc="train", leave=False)):
            data, gt = data.cuda(), gt.cuda()
            data = pixel_patch_featuring(data)
            data = rearrange(data, "b (h w) c -> (b h w) c")
            gt = rearrange(patch_gt_pooling(gt), "b (h w) c -> b c h w", h=52)
            normal_gt = (ignore_neighbor(gt) == 0)
            labels = -1 * torch.ones_like(gt, dtype=torch.int32)
            labels[normal_gt] = 0
            labels[gt > 0] = 1
            dists = model(data, gt)

            pull_loss = torch.mean(torch.max(dists[labels == 0] - model.r, 0))
            push_loss = torch.mean(torch.max(model.r - dists[labels == 1] + args.alpha, 0))

            loss = args.reg_lambda * pull_loss + push_loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_pull_loss += pull_loss.item()
            total_push_loss += push_loss.item()
            total_step += data.shape[0]

        writer.add_scalar("Loss/Train Total", total_loss / total_step, epoch)
        writer.add_scalar("Loss/Train Pull", total_pull_loss / total_step, epoch)
        writer.add_scalar("Loss/Train Push", total_push_loss / total_step, epoch)

        model.eval()
        total_loss = 0.0
        total_push_loss = 0.0
        total_pull_loss = 0.0
        total_step = 0

        y_true = []
        y_pred = []
        with torch.no_grad():
            # (1) evaluate ID classification
            correct, total = 0, 0
            for data, labels in tqdm(test_id_loader, desc="Test-Classification", leave=False):
                data, labels = data.cuda(), labels.cuda()
                dists = model(data) 
                scores = - dists + model.alphas

                label_mask = torch.zeros(labels.size(0), model.num_classes).cuda().scatter_(1, labels.unsqueeze(dim=1), 1)

                pull_loss = torch.mean(torch.sum(torch.mul(label_mask, dists), dim=1))
                push_loss = ce_loss(scores, labels)
                loss = args.reg_lambda * pull_loss + push_loss 
                total_loss += loss.item()
                total_pull_loss += pull_loss.item()
                total_push_loss += push_loss.item()
                total_step += data.shape[0]

                _, predicted = torch.max(scores, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                y_pred.append(predicted.cpu().detach().numpy())
                y_true.append(labels.cpu().detach().numpy())
            accuracy = 100 * correct / total
            idacc_list.append(accuracy)
            if accuracy > max_acc:
                max_acc = accuracy
                loss_str = ""
                if total_loss < min_loss:
                    min_loss = total_loss
                    loss_str = f"(Loss = {total_loss:.6f})"
                logger.info(f"Best model saved at {epoch} epoch(Accuracy = {accuracy:.2f}){loss_str}")
                torch.save(model.state_dict(), os.path.join(outdir, "best.pt"))
                torch.save(model.state_dict(), os.path.join(outdir, f"{epoch}.pt"))
            elif total_loss < min_loss:
                min_loss = total_loss
                logger.info(f"Best model saved at {epoch} epoch(Loss = {total_loss:.6f})")
                torch.save(model.state_dict(), os.path.join(outdir, f"{epoch}.pt"))


        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        writer.add_scalar("Eval/InDist acc", accuracy, epoch)
        writer.add_scalar("Eval/InDist balance-acc", balanced_accuracy, epoch)
        writer.add_scalar("Loss/Valid Total", total_loss / total_step, epoch)
        writer.add_scalar("Loss/Valid Pull", total_pull_loss / total_step, epoch)
        writer.add_scalar("Loss/Valid Push", total_push_loss / total_step, epoch)
        # centroid_dist = (
        #     torch.cdist(
        #         model.centers[0,:].reshape(1,1,-1),
        #         model.centers[1,:].reshape(1,1,-1)
        #     ).squeeze().item()
        # )
        # writer.add_scalar("Param/D(mu_0, mu_1)", centroid_dist, epoch)
        writer.add_scalar("Param/Sigma_0", model.logsigmas[0], epoch)
        # writer.add_scalar("Param/Sigma_1", model.logsigmas[1], epoch)
        writer.add_scalar("Param/alpha_0", model.alphas[0], epoch)
        # writer.add_scalar("Param/alpha_1", model.alphas[1], epoch)

        writer.flush()

    logger.info('== results ==')
    logger.info('The best ID accuracy on "{idset:s}" test samples : {val:6.2f}'.format(idset=args.dataset, val=best_idacc))

    writer.close()


if __name__ == '__main__':
    with logging_redirect_tqdm():
        main()
