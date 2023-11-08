import torch
import torch.nn as nn
import torch.nn.functional as TF

class VAE(nn.Module):
    def __init__(self, in_channel, hidden_channels, latent_dim, in_shape):
        super(VAE,self).__init__()

        self.in_shape = in_shape
        self.latent_dim = latent_dim
        self.in_channels = [in_channel] + hidden_channels
        self.padding_mode = "replicate"

        self.latent_shape = [s//(2**len(hidden_channels)) for s in in_shape]
        self.l1 = int(self.in_channels[-1] * torch.prod(torch.tensor(self.latent_shape)))
        self.l2 = int(self.latent_dim)

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
        # self.encoder = nn.Sequential(
        #     # input is (nc) x 16 x 16
        #     nn.Conv2d(nc, ndf, 4, 2, 1, padding_mode=self.padding_mode),
        #     nn.ReLU(0.2, inplace=True),
        #     # state size. (ndf) x 8 x 8
        #     nn.Conv2d(ndf, ndf * 2, 4, 2, 1, padding_mode=self.padding_mode),
        #     nn.BatchNorm2d(ndf * 2),
        #     nn.ReLU(0.2, inplace=True),
        #     nn.Dropout2d(0.5),
        #     # state size. (ndf*2) x 4 x 4
        #     nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, padding_mode=self.padding_mode),
        #     nn.BatchNorm2d(ndf * 4),
        #     nn.ReLU(0.2, inplace=True),
        #     nn.Dropout2d(0.5),
        #     # state size. (ndf*4) x 2 x 2
        #     nn.Conv2d(ndf * 4, 64, 2, 1, 0, padding_mode=self.padding_mode),
        #     # nn.BatchNorm2d(1024),
        #     nn.ReLU(0.2, inplace=True),
        #     # nn.Sigmoid()
        # )
        
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
                nn.LeakyReLU(),
            ])
        layer_list.extend([
            nn.ConvTranspose2d(
                in_channels=self.in_channels[1],
                out_channels=self.in_channels[1],
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(self.in_channels[1]),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=self.in_channels[1],
                out_channels=self.in_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=self.padding_mode
            ),
        ])
        self.decoder = nn.Sequential(*layer_list)
        # self.decoder = nn.Sequential(
        #     # input is Z, going into a convolution
        #     nn.ConvTranspose2d(64, ngf * 4, 2, 1, 0, padding_mode=self.padding_mode),
        #     nn.BatchNorm2d(ngf * 4),
        #     nn.ReLU(True),
        #     # state size. (ngf*8) x 4 x 4
        #     nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, padding_mode=self.padding_mode),
        #     nn.BatchNorm2d(ngf * 2),
        #     nn.ReLU(True),
        #     # state size. (ngf*4) x 8 x 8
        #     nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, padding_mode=self.padding_mode),
        #     nn.BatchNorm2d(ngf),
        #     nn.ReLU(True),
        #     # state size. (ngf*2) x 16 x 16
        #     nn.ConvTranspose2d(ngf, nc, 4, 2, 1, padding_mode=self.padding_mode),
        #     # nn.BatchNorm2d(ngf),
        #     # nn.ReLU(True),
        #     # state size. (ngf) x 32 x 32
        #     # nn.ConvTranspose2d(	 ngf,	   nc, 4, 2, 1, bias=False),
        #     # nn.Tanh()
        #     # nn.Sigmoid()
        #     # state size. (nc) x 64 x 64
        # )
        
        self.fc1 = nn.Linear(self.l1, self.l2)
        self.mu = nn.Linear(self.l2, latent_dim)
        self.logvar = nn.Linear(self.l2, latent_dim)
        
        self.fc3 = nn.Linear(latent_dim, self.l2)
        self.fc4 = nn.Linear(self.l2, self.l1)
        
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def encode(self, x):
        conv = self.encoder(x)
        h1 = self.relu(self.fc1(conv.view(-1, self.l1)))
        return self.mu(h1), self.logvar(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        deconv_input = deconv_input.view(-1,self.in_channels[-1],*self.latent_shape)
        return self.decoder(deconv_input)
    
    def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor):
        std = logvar.mul(0.5).exp_()
        # if torch.any(torch.isinf(std)):
        #     std[torch.isinf(std)] = torch.tensor((0.5,), device=std.device).exp()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
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
        # return torch.mean(((x - reco_x + 1e-9) ** 2), dim=1)
        return torch.sum((reco_x - x) ** 2, dim=1)

    def RE(self, reco_x, x):
        # return torch.mean(self.ch_re(reco_x, x))
        return torch.mean(torch.sum(self.ch_re(reco_x, x), dim=(1,2)))
        # return TF.binary_cross_entropy(reco_x, x)

    def KLD(self, mu, logvar):
        kld = 1 + logvar - (mu + 1e-9).pow(2) - (logvar + 1e-9).exp()
        # if torch.any(torch.isinf(kld)):
        #     kld[torch.isinf(kld)] = 0
        return torch.mean(-0.5*torch.sum(kld, dim=1))

    def loss_function(self, reco_x, x, mu, logvar):
        return self.RE(reco_x, x) + self.KLD(mu, logvar)