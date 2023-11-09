import os, argparse, datetime
import torch
from torchinfo import summary
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np

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
parser.add_argument('--dataset', required=True, help='gas | shuttle | drive | mnist')
parser.add_argument('--net_type', required=True, help='mlp')
parser.add_argument('--datadir', default='./table_data/', help='path to dataset')
parser.add_argument('--outdir', default='./output/', help='folder to output results')
parser.add_argument('--oodclass_idx', type=int, default=0, help='index of the OOD class')
parser.add_argument('--batch_size', type=int, default=200, help='batch size for data loader')
parser.add_argument('--latent_size', type=int, default=128, help='dimension size for latent representation') 
parser.add_argument('--num_layers', type=int, default=3, help='the number of hidden layers in MLP')
parser.add_argument('--num_folds', type=int, default=5, help='the number of cross-validation folds')
parser.add_argument('--num_epochs', type=int, default=10, help='the number of epochs for training sc-layers')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate of Adam optimizer')
parser.add_argument('--reg_lambda', type=float, default=1.0, help='regularization coefficient')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')

args = parser.parse_args()

set_logger("log.yaml", args.outdir)
logger.info(args)

def main():
    outdir = os.path.join(args.outdir, args.net_type + '_' + args.dataset)

    if os.path.isdir(outdir) == False:
        os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(args.gpu)

    best_idacc_list, best_oodacc_list = [], []
    for fold_idx in range(args.num_folds):
        
        train_loader, test_id_loader, test_ood_loader = get_table_data(args.batch_size, args.datadir, args.dataset, args.oodclass_idx, fold_idx)

        if test_ood_loader is not None:
            test_ood = True
        else:
            test_ood = False
        test_ood = False
        logdir=f"runs/{datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')}_{os.path.basename(args.outdir)}_{fold_idx}"
        writer = SummaryWriter(log_dir=logdir)
        max_acc = 0
        min_loss = 1e10

        if args.dataset == 'gas':
            num_classes, num_features = 6, 128
        elif args.dataset == 'drive':
            num_classes, num_features = 11, 48
        elif args.dataset == 'shuttle':
            num_classes, num_features = 7, 9
        elif args.dataset == 'mnist':
            num_classes, num_features = 10, 784
        elif args.dataset == "dac_1class":
            num_classes, num_features = 1, 1536
        elif args.dataset == "dac_2class":
            num_classes, num_features = 2, 1536
        elif args.dataset == "dac_3class":
            num_classes, num_features = 3, 1536
        elif args.dataset == "vae_features":
            num_classes, num_features = 2, 128

        model = models.MLP_DeepMCDD(num_features, [256, 128, 64, 16], num_classes=num_classes)
        model.cuda()
        logger.info(summary(model, (1, 1, 16, 16), col_names=["input_size", "output_size", "kernel_size"]))

        ce_loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        idacc_list, oodacc_list = [], []
        for epoch in trange(args.num_epochs, desc="epochs", leave=False):
            model.train()
            total_loss = 0.0
            total_push_loss = 0.0
            total_pull_loss = 0.0
            total_step = 0

            for i, (data, labels) in enumerate(tqdm(train_loader, desc="train", leave=False)):
                data, labels = data.cuda(), labels.cuda()
                dists = model(data) 
                scores = - dists + model.alphas

                label_mask = torch.zeros(labels.size(0), model.num_classes).cuda().scatter_(1, labels.unsqueeze(dim=1), 1)

                pull_loss = torch.mean(torch.sum(torch.mul(label_mask, dists), dim=1))
                push_loss = ce_loss(scores, labels)
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

                if test_ood:
                    # (2) evaluate OOD detection
                    compute_confscores(model, test_id_loader, outdir, True)
                    compute_confscores(model, test_ood_loader, outdir, False)
                    oodacc_list.append(compute_metrics(outdir))

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
            
            for i in range(model.num_classes):
                writer.add_scalar(f"Param/Sigma_{i}", model.logsigmas[i], epoch)
                writer.add_scalar(f"Param/alpha_{i}", model.alphas[i], epoch)

            writer.flush()
        best_idacc = max(idacc_list)
        if test_ood:
            best_oodacc = oodacc_list[idacc_list.index(best_idacc)]
        
        logger.info('== {fidx:1d}-th fold results =='.format(fidx=fold_idx+1))
        logger.info('The best ID accuracy on "{idset:s}" test samples : {val:6.2f}'.format(idset=args.dataset, val=best_idacc))
        if test_ood:
            logger.info('The best OOD accuracy on "{oodset:s}" test samples :'.format(oodset=args.dataset+'_'+str(args.oodclass_idx)))
            print_ood_results(best_oodacc)

        best_idacc_list.append(best_idacc)
        if test_ood:
            best_oodacc_list.append(best_oodacc)
        writer.close()

    logger.info('== Final results ==')
    logger.info('The best ID accuracy on "{idset:s}" test samples : {mean:6.2f} ({std:6.3f})'.format(idset=args.dataset, mean=np.mean(best_idacc_list), std=np.std(best_idacc_list)))
    if test_ood:
        logger.info('The best OOD accuracy on "{oodset:s}" test samples :'.format(oodset='class_'+str(args.oodclass_idx)))
        print_ood_results_total(best_oodacc_list)



if __name__ == '__main__':
    with logging_redirect_tqdm():
        main()
