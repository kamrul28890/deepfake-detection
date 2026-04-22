from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils


class FrameDataset(Dataset):
    def __init__(self, image_paths: Iterable[str | Path], image_size: int = 64):
        self.image_paths = [Path(path) for path in image_paths]
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 128, image_channels: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


class Discriminator(nn.Module):
    def __init__(self, image_channels: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image).view(-1)


@dataclass
class GANResult:
    generator_path: Path
    sample_grid_path: Path


def train_dcgan(
    image_paths: Iterable[str | Path],
    output_dir: str | Path,
    epochs: int = 1,
    batch_size: int = 32,
    latent_dim: int = 128,
    device: str | None = None,
) -> GANResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FrameDataset(image_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    for _ in range(epochs):
        for real_images in loader:
            real_images = real_images.to(device)
            batch_size_actual = real_images.size(0)
            real_labels = torch.ones(batch_size_actual, device=device)
            fake_labels = torch.zeros(batch_size_actual, device=device)

            noise = torch.randn(batch_size_actual, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)

            discriminator.zero_grad(set_to_none=True)
            real_loss = criterion(discriminator(real_images), real_labels)
            fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            generator.zero_grad(set_to_none=True)
            g_loss = criterion(discriminator(fake_images), real_labels)
            g_loss.backward()
            optimizer_g.step()

    generator_path = output_dir / "dcgan_generator.pt"
    torch.save(generator.state_dict(), generator_path)

    with torch.no_grad():
        sample_noise = torch.randn(16, latent_dim, 1, 1, device=device)
        sample_images = generator(sample_noise).cpu()
        sample_grid_path = output_dir / "dcgan_samples.png"
        utils.save_image(sample_images, sample_grid_path, nrow=4, normalize=True, value_range=(-1, 1))

    return GANResult(generator_path=generator_path, sample_grid_path=sample_grid_path)
