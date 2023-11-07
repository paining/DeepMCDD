import torch
import torch.nn as nn
import torch.nn.functional as TF

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv1 = nn.Conv2d(in_channels, 2*in_channels, kernel_size=(1,1))
        self.conv3 = nn.Conv2d(in_channels, 2*in_channels, kernel_size=(3,3), padding=(1,1))
        self.conv5 = nn.Conv2d(in_channels, 2*in_channels, kernel_size=(5,5), padding=(2,2))
        self.outconv = nn.Conv2d(6*in_channels, out_channels, kernel_size=(1,1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = self.conv5(x)
        x = x + torch.cat(x1, x2, x3, dim=1)
        return self.outconv(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode="replicate"):
        super(ConvBlock,self).__init__()
        self.in_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1,1), padding_mode=padding_mode
        )
        self.layer = nn.Sequential(
            nn.Conv2d(out_channels, 2*out_channels, kernel_size=(1,1)),
            nn.BatchNorm2d(2*out_channels),
            nn.ReLU(),
            nn.Conv2d(
                2*out_channels, out_channels,
                kernel_size=(3,3), padding=(1,1), padding_mode=padding_mode
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.out_conv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(4,4), stride=(2,2), padding=(1,1), padding_mode=padding_mode
        )

    def forward(self, x):
        x = self.in_conv(x)
        out = self.layer(x)
        return self.out_conv(out + x)

class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode="replicate"):
        super(DeConvBlock,self).__init__()
        self.in_conv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=(4,4), stride=(2,2), padding=(1,1)
        )
        self.layer = nn.Sequential(
            nn.Conv2d(out_channels, 2*out_channels, kernel_size=(1,1)),
            nn.BatchNorm2d(2*out_channels),
            nn.ReLU(),
            nn.Conv2d(
                2*out_channels, out_channels,
                kernel_size=(3,3), padding=(1,1), padding_mode=padding_mode
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(1,1))

    def forward(self, x):
        x = self.in_conv(x)
        out = self.layer(x)
        return self.out_conv(out + x)

class VAE(nn.Module):
    def __init__(self, in_channel, hidden_channels, latent_dim, in_shape):
        super(VAE,self).__init__()

        self.in_shape = in_shape
        self.latent_dim = latent_dim
        self.in_channels = [in_channel] + hidden_channels

        self.selu = nn.SELU()
        self.relu = nn.ReLU()

        self.encoderblock = nn.Sequential(*[
            ConvBlock(self.in_channels[i], self.in_channels[i+1])
            for i in range(len(self.in_channels) - 1)
        ])

        self.latent_shape = [s//(2**len(self.encoderblock)) for s in in_shape]
        self.l1 = int(self.in_channels[-1] * torch.prod(torch.tensor(self.latent_shape)))
        # self.l2 = int(self.in_channels[-1] * torch.sqrt(torch.prod(torch.tensor(self.latent_shape))))
        self.l2 = int(self.in_channels[-1])

        self.decoderblock = nn.Sequential(*[
            DeConvBlock(self.in_channels[i], self.in_channels[i-1])
            for i in range(len(self.in_channels)-1, 0, -1)
        ])

        self.en_fc = nn.Linear(self.l1, self.l2)
        self.mu = nn.Linear(self.l2, self.latent_dim)
        self.logvar = nn.Linear(self.l2, self.latent_dim)

        self.de_fc = nn.Sequential(
            nn.Linear(self.latent_dim, self.l2),
            nn.ReLU(),
            nn.Linear(self.l2, self.l1),
        )

        # self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        out = self.encoderblock(x)
        out = out.flatten()
        return out

    def decoder(self, z):
        out = self.decoderblock(z)
        # return self.sigmoid(out)
        return out

    def encode(self, x):
        h = self.encoder(x)
        h = self.relu(self.en_fc(h.reshape(-1, self.l1)))

        return self.mu(h), self.logvar(h)

    def decode(self, z):
        h = self.de_fc(z)
        return self.decoder(h.reshape(-1, self.in_channels[-1], *self.latent_shape))
    
    def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = torch.autograd.Variable(eps)
        
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        # std, mean = torch.std_mean(x, dim=(1,2,3))
        # x = (x - mean)/std
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        # decoded = (decoded + mean)*std
        
        return decoded, mu, logvar
    
    def ch_re(self, reco_x, x):
        
        return torch.mean(((x - reco_x + 1e-9) ** 2), dim=-1)
    
    def RE(self, reco_x, x):
        
        # return torch.mean(self.ch_re(reco_x, x))
        return torch.sum(self.ch_re(reco_x, x))
    
    def KLD(self, mu, logvar):
        
        return -0.5*torch.sum(1 + logvar - (mu + 1e-9).pow(2) - (logvar + 1e-9).exp())
    
    def loss_function(self, reco_x, x, mu, logvar):
        
        return self.RE(reco_x, x) + self.KLD(mu, logvar)