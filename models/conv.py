import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class Conv_DeepMCDD(nn.Module):
    def __init__(self, input_channel, hidden_channeles, num_classes, kernel_size, padding_mode=None):
        super(Conv_DeepMCDD, self).__init__()
        self.kernel_size = kernel_size
        if padding_mode is not None:
            self.padding_size = kernel_size//2
            self.padding_mode = padding_mode
        else:
            self.padding_size = 0
        self.input_channel = input_channel
        self.hidden_channeles = hidden_channeles[:-1]
        self.latent_size = hidden_channeles[-1]
        self.num_classes = num_classes

        self.centers = torch.nn.Parameter(torch.zeros([num_classes, self.latent_size]), requires_grad=True)
        self.alphas = torch.nn.Parameter(torch.zeros(num_classes), requires_grad=True)
        self.logsigmas = torch.nn.Parameter(torch.zeros(num_classes), requires_grad=True)
        
        self.build_fe()
        self.init_fe_weights()

    def build_fe(self):
        layers = []
        layer_sizes = [self.input_channel] + self.hidden_channeles

        if self.padding_size != 0:
            for i in range(len(layer_sizes)-1):
                layers.append(nn.Conv2d(
                    layer_sizes[i],
                    layer_sizes[i+1],
                    self.kernel_size,
                    padding=self.padding_size,
                    padding_mode=self.padding_mode,
                    )
                )
                layers.append(nn.BatchNorm2d(layer_sizes[i+1]))
                layers.append(nn.LeakyReLU())
                # layers.append(nn.ReLU())
        else:
            for i in range(len(layer_sizes)-1):
                layers.append(nn.Conv2d(
                    layer_sizes[i],
                    layer_sizes[i+1],
                    self.kernel_size,
                    )
                )
                layers.append(nn.BatchNorm2d(layer_sizes[i+1]))
                layers.append(nn.LeakyReLU())
                # layers.append(nn.ReLU())
        layers.append(nn.AdaptiveAvgPool2d((1,1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(layer_sizes[-1], 64))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(64, self.latent_size))
        self.layers = nn.ModuleList(layers)

    def init_fe_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.centers)
        nn.init.zeros_(self.alphas)
        nn.init.zeros_(self.logsigmas)

    def _forward(self, x):
        if len(x.shape) == 2 or (len(x.shape) == 3 and x.shape[1] == 1):
            x = x.reshape(x.shape[0], 1, 8, 8)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def forward(self, x):
        out = self._forward(x)
        out = out.unsqueeze(dim=1).repeat([1, self.num_classes, 1])
        scores = torch.sum((out - self.centers)**2, dim=2) / 2 / torch.exp(2 * F.relu(self.logsigmas)) + self.latent_size * F.relu(self.logsigmas)
        return scores

class Conv_DeepMCDD_oneclass(nn.Module):
    def __init__(self, input_channel, hidden_channeles, kernel_size, padding_mode=None):
        super(Conv_DeepMCDD_oneclass, self).__init__()
        self.kernel_size = kernel_size
        if padding_mode is not None:
            self.padding_size = kernel_size//2
            self.padding_mode = padding_mode
        else:
            self.padding_size = 0
        self.input_channel = input_channel
        self.hidden_channeles = hidden_channeles[:-1]
        self.latent_size = hidden_channeles[-1]

        self.r = torch.nn.Parameter(torch.ones((1,)), requires_grad=True)
        
        self.build_fe()
        self.init_fe_weights()

    def build_fe(self):
        layers = []
        layer_sizes = [self.input_channel] + self.hidden_channeles

        if self.padding_size != 0:
            for i in range(len(layer_sizes)-1):
                layers.append(nn.Conv2d(
                    layer_sizes[i],
                    layer_sizes[i+1],
                    self.kernel_size,
                    padding=self.padding_size,
                    padding_mode=self.padding_mode,
                    )
                )
                layers.append(nn.BatchNorm2d(layer_sizes[i+1]))
                layers.append(nn.LeakyReLU())
                # layers.append(nn.ReLU())
        else:
            for i in range(len(layer_sizes)-1):
                layers.append(nn.Conv2d(
                    layer_sizes[i],
                    layer_sizes[i+1],
                    self.kernel_size,
                    )
                )
                layers.append(nn.BatchNorm2d(layer_sizes[i+1]))
                layers.append(nn.LeakyReLU())
                # layers.append(nn.ReLU())
        layers.append(nn.AdaptiveAvgPool2d((1,1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(layer_sizes[-1], 64))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(64, self.latent_size))
        self.layers = nn.ModuleList(layers)

    def init_fe_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


    def _forward(self, x):
        if len(x.shape) == 2 or (len(x.shape) == 3 and x.shape[1] == 1):
            x = x.reshape(x.shape[0], 1, 8, 8)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def forward(self, x):
        # x : (B, 64), label : (B, 1)
        out = self._forward(x) # B, C, H, W
        return out
        # out = rearrange(out, "b c h w -> (b h w) c")
        # mean = torch.mean(out[label == 0], dim=0)
        # dist = torch.cdist(out.reshape(1, *out.shape), mean.reshape(1, 1, *mean.shape))
        # return dist

class MultiScaleConv_DeepMCDD(nn.Module):
    def __init__(self, input_channel, hidden_channeles, num_classes, kernel_size):
        super(Conv_DeepMCDD, self).__init__()
        self.kernel_size = kernel_size
        # self.padding_size = kernel_size//2
        self.input_channel = input_channel
        self.hidden_channeles = hidden_channeles[:-1]
        self.latent_size = hidden_channeles[-1]
        self.num_classes = num_classes

        self.centers = torch.nn.Parameter(torch.zeros([num_classes, self.latent_size]), requires_grad=True)
        self.alphas = torch.nn.Parameter(torch.zeros(num_classes), requires_grad=True)
        self.logsigmas = torch.nn.Parameter(torch.zeros(num_classes), requires_grad=True)
        
        self.build_fe()
        self.init_fe_weights()

    def build_fe(self):
        layers = []
        layer_sizes = [self.input_channel] + self.hidden_channeles
       
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Conv2d(
                layer_sizes[i],
                layer_sizes[i+1],
                self.kernel_size,
                # padding=self.padding_size,
                padding_mode="replicate",
                )
            )
            layers.append(nn.BatchNorm2d(layer_sizes[i+1]))
            layers.append(nn.LeakyReLU())
            # layers.append(nn.ReLU())
        layers.append(nn.AdaptiveAvgPool2d((1,1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(layer_sizes[-1], self.latent_size))
        self.layers = nn.ModuleList(layers)

    def init_fe_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.centers)
        nn.init.zeros_(self.alphas)
        nn.init.zeros_(self.logsigmas)

    def _forward(self, x):
        if len(x.shape) == 2 or (len(x.shape) == 3 and x.shape[1] == 1):
            x = x.reshape(x.shape[0], 1, 8, 8)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def forward(self, x):
        out = self._forward(x)
        out = out.unsqueeze(dim=1).repeat([1, self.num_classes, 1])
        scores = torch.sum((out - self.centers)**2, dim=2) / 2 / torch.exp(2 * F.relu(self.logsigmas)) + self.latent_size * F.relu(self.logsigmas)
        return scores
