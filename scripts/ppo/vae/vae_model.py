import os
import torch as T
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dims: tuple[int, int, int], latent_dim: int, device: str = "cpu"):
        super().__init__()
        assert len(input_dims) == 3, "input dimensions must be (C, W, H)"
        c, w, h = input_dims

        self.conv = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.mu_linear = nn.Linear(256 * 4 * 4, latent_dim)
        self.logvar_linear = nn.Linear(256 * 4 * 4, latent_dim)

        self.device = device
        self.to(device)

    def forward(self, x: T.Tensor) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        out = self.conv(x)
        out = out.view(out.size(0), -1)

        mu = self.mu_linear(out)
        log_var = self.logvar_linear(out)
        std = T.exp(0.5 * log_var)
        eps = T.randn_like(std)
        z = mu + eps * std
        return z, mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_channels: int = 3, device: str = "cpu"):
        super().__init__()
        self.linear_layer = nn.Linear(latent_dim, 256 * 4 * 4)

        self.deconv = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # for normalized outputs in [0,1]
        )

        self.device = device
        self.to(device)

    def forward(self, x: T.Tensor) -> T.Tensor:
        out = self.linear_layer(x)
        out = out.view(-1, 256, 4, 4)
        return self.deconv(out)


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dims: tuple[int, int, int], latent_dim: int, device: str = "cpu"):
        super().__init__()
        self.encoder = Encoder(input_dims, latent_dim, device)
        self.decoder = Decoder(latent_dim, output_channels=input_dims[0], device=device)
        self.device = device
        self.to(device)

    def forward(self, x: T.Tensor) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        z, mu, logvar = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    def save(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        T.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
        }, os.path.join(save_path, "vae_checkpoint.pth"))

    def load(self, load_path: str):
        checkpoint = T.load(load_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.to(self.device)


if __name__ == "__main__":
    x = T.randn(1, 3, 64, 64)
    vae = VariationalAutoEncoder((3, 64, 64), latent_dim=128)
    recon, mu, logvar = vae(x)
    print("Reconstruction shape:", recon.shape)
