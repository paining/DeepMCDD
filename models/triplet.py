import torch
import torch.nn as nn
import torch.nn.functional as TF
from pytorch_metric_learning import miners, losses

class ConvEocoder(nn.Module):
    def __init__(self, in_channel, hidden_channels):
        super(ConvEocoder,self).__init__()

        self.in_channels = [in_channel] + hidden_channels
        self.padding_mode = "replicate"


        layer_list = []
        for i in range(len(self.in_channels) - 1):
            layer_list.extend([
                nn.Conv2d(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i+1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    padding_mode=self.padding_mode
                ),
                nn.BatchNorm2d(self.in_channels[i+1]),
                nn.LeakyReLU(),
                nn.Dropout(0.1)
            ])
        self.encoder = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.encoder(x)

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


class TripletModel(nn.Module):
    def __init__(
            self,
            in_channel,
            hidden_channels,
            latent_dim,
            in_shape,
            anchor_id,
            margin,
            sampling_type="semihard"
        ):
        super(TripletModel, self).__init__()

        self.in_channels = [in_channel] + hidden_channels
        self.in_shape = in_shape
        self.latent_dim = latent_dim
        self.latent_shape = [s//(2**len(hidden_channels)) for s in in_shape]
        self.l1 = int(self.in_channels[-1] * torch.prod(torch.tensor(self.latent_shape)))
        self.l2 = int(self.latent_dim)

        self.margin = margin
        if isinstance(anchor_id, list): self.anchor_id = anchor_id
        elif isinstance(anchor_id, int): self.anchor_id = [anchor_id]
        else: raise ValueError("Anchor id must be list or integer")

        if sampling_type == "semihard":
            self.make_triplet = make_triplet_softhard
        else:
            self.make_triplet = make_triplet_all

        self.layers = torch.nn.Sequential(
            ConvEocoder(in_channel, hidden_channels),
            nn.Flatten(1, -1),
            nn.Linear(self.l1, self.l2, bias=False),
            nn.Linear(self.l2, self.latent_dim, bias=False)
        )

    def forward(self, x):
        return self.layers(x)

    def loss_function(self, x, y):
        emb = self.forward(x)
        total_loss = []
        for id in self.anchor_id:
            ret = self.make_triplet(emb, y, id, self.margin)
            if ret is not None:
                anchor, positive, negative = ret
                total_loss.append(TF.triplet_margin_loss(anchor, positive, negative))
        if not total_loss: return None
        total_loss = torch.sum(torch.stack(total_loss))
        return total_loss