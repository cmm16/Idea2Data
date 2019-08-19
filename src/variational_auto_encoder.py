import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, image_input):
        print("flatten", print(image_input.size(0)))
        return image_input.view(image_input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, image_input):
        return image_input.view(image_input.size(0), image_input.size(1), 1, 1)


class VAE(nn.Module):
    def __init__(self, input_size, image_channels=3, z_dim=36):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        h_dim = self.determine_flattened_size(input_size)
        print("h_dim", h_dim)
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=8, stride=4),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        print(h.shape)
        mu = self.fc1(h)
        logvar = self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        print(h.shape)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        print("decode", z.shape)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def determine_flattened_size(self, input_size):
        flatten_dim = 0
        for layer in self.encoder:
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                # using equation ((N_1 + 2P_1 - F_1)/S_1 + 1) X ((N_2 + 2P_2 - F_2)/S_2 + 1) X C
                input_size = ((input_size[0] + (2 * layer.padding[0]) - layer.kernel_size[0]) // layer.stride[0] + 1,
                              (input_size[1] + (2 * layer.padding[1]) - layer.kernel_size[1]) // layer.stride[1] + 1)
                flatten_dim = layer.out_channels * input_size[0] * input_size[1]
        return flatten_dim


