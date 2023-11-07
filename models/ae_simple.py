import torch
import torch.nn as nn
import torch.nn.functional as TF

class AE(nn.Module):
    def __init__(self, in_channel, hidden_channels, latent_dim, in_shape):
        super(AE,self).__init__()

        self.in_shape = in_shape
        self.latent_dim = latent_dim
        self.in_channels = [in_channel] + hidden_channels
        self.padding_mode = "replicate"

        self.latent_shape = [s//(2**len(hidden_channels)) for s in in_shape]
        self.l1 = int(self.in_channels[-1] * torch.prod(torch.tensor(self.latent_shape)))
        self.l2 = int(self.in_channels[-1])

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
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        self.encoder = nn.Sequential(*layer_list)

        layer_list = []
        for i in range(len(self.in_channels) - 1, 1, -1):
            layer_list.extend([
                nn.ConvTranspose2d(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i-1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(self.in_channels[i-1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        layer_list.append(
            nn.ConvTranspose2d(
                in_channels=self.in_channels[1],
                out_channels=self.in_channels[0],
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )
        layer_list.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layer_list)

        self.fc1 = nn.Linear(self.l1, self.l2)
        self.fc2 = nn.Linear(self.l2, latent_dim)
        
        self.fc3 = nn.Linear(latent_dim, self.l2)
        self.fc4 = nn.Linear(self.l2, self.l1)
        
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def encode(self, x):
        conv = self.encoder(x)
        h1 = self.relu(self.fc1(conv.view(-1, self.l1)))
        return self.fc2(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        deconv_input = deconv_input.view(-1,self.in_channels[-1],*self.latent_shape)
        return self.decoder(deconv_input)

    def forward(self, x):
        z = self.encode(x)
        decoded = self.decode(z)

        return decoded

    def ch_re(self, reco_x, x):
        return torch.mean(((x - reco_x + 1e-9) ** 2), dim=-2)

    def RE(self, reco_x, x):
        # return torch.mean(self.ch_re(reco_x, x))
        return torch.mean(torch.sum(self.ch_re(reco_x, x), dim=(1,2)))
        # return TF.binary_cross_entropy(reco_x, x)

    def loss_function(self, reco_x, x, mu, logvar):
        return self.RE(reco_x, x)