import torch
import torch.nn as nn
import torch.nn.functional as F


def get_act(name):
    if name == "relu":       return nn.ReLU()
    if name == "leaky_relu": return nn.LeakyReLU(0.2)
    if name == "elu":        return nn.ELU()
    raise ValueError(f"Unknown activation: {name}")


class ResBlock(nn.Module):

    def __init__(self, channels, filter_size, act_name):
        super().__init__()
        pad = filter_size // 2

        self.conv1 = nn.Conv2d(channels, channels, filter_size, padding=pad)
        self.bn1   = nn.BatchNorm2d(channels)
        self.act1  = get_act(act_name)

        self.conv2 = nn.Conv2d(channels, channels, filter_size, padding=pad)
        self.bn2   = nn.BatchNorm2d(channels)
        self.act2  = get_act(act_name)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + x
        out = self.act2(out)
        return out


class UpsampleBlock(nn.Module):

    def __init__(self, in_ch, out_ch, filter_size):
        super().__init__()
        pad = filter_size // 2
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, filter_size, padding=pad)

    def forward(self, x):
        return self.conv(self.up(x))


class VAE(nn.Module):

    def __init__(self, latent_dim=128, num_classes=2, filter_size=3,
                 num_layers=3, activation="relu", decoder_type="deconv",
                 num_res_blocks=1):
        super().__init__()

        self.latent_dim  = latent_dim
        self.num_classes = num_classes

        pad          = filter_size // 2
        enc_channels = [3] + [32 * (2 ** i) for i in range(num_layers)]

        enc_layers = []
        for i in range(num_layers):
            enc_layers.append(nn.Conv2d(enc_channels[i], enc_channels[i+1],
                                        filter_size, stride=2, padding=pad))
            enc_layers.append(nn.BatchNorm2d(enc_channels[i+1]))
            enc_layers.append(get_act(activation))
            for _ in range(num_res_blocks):
                enc_layers.append(ResBlock(enc_channels[i+1], filter_size, activation))

        self.encoder        = nn.Sequential(*enc_layers)
        self.final_channels = enc_channels[-1]
        self.spatial        = 64 // (2 ** num_layers)
        flat_dim            = self.final_channels * self.spatial * self.spatial

        self.fc_mu     = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim + num_classes, flat_dim)

        dec_channels = [self.final_channels // (2 ** i) for i in range(num_layers)] + [3]

        dec_layers = []
        for i in range(num_layers):
            is_last = (i == num_layers - 1)

            if decoder_type == "deconv":
                dec_layers.append(
                    nn.ConvTranspose2d(dec_channels[i], dec_channels[i+1],
                                      filter_size, stride=2, padding=pad, output_padding=1)
                )
            else:
                dec_layers.append(UpsampleBlock(dec_channels[i], dec_channels[i+1], filter_size))

            if not is_last:
                dec_layers.append(nn.BatchNorm2d(dec_channels[i+1]))
                dec_layers.append(get_act(activation))
                for _ in range(num_res_blocks):
                    dec_layers.append(ResBlock(dec_channels[i+1], filter_size, activation))

        dec_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h      = self.encoder(x).flatten(1)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        label_oh = F.one_hot(labels, self.num_classes).float()
        x        = torch.cat([z, label_oh], dim=1)
        x        = self.fc_decode(x)
        x        = x.view(x.size(0), self.final_channels, self.spatial, self.spatial)
        return self.decoder(x)

    def forward(self, x, labels):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z, labels)
        return recon, mu, logvar

    def generate(self, label, n, device):
        self.eval()
        with torch.no_grad():
            z      = torch.randn(n, self.latent_dim).to(device)
            labels = torch.full((n,), label, dtype=torch.long).to(device)
            imgs   = self.decode(z, labels)
        return imgs


def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total      = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss