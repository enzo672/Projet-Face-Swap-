import sys
import os
import datetime
import time
from random import choice
from glob import glob
from pathlib import Path
import platform
import numpy as np
import cv2
import torchvision.utils
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

plt.style.use('dark_background')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------------------------------------------------
# ENVIRONNEMENT GRAPHIQUE
# ------------------------------------------------------------
def can_display():
    if platform.system() == "Linux":
        return "DISPLAY" in os.environ and bool(os.environ["DISPLAY"])
    elif platform.system() in ["Darwin", "Windows"]:
        return True
    return False

HAS_DISPLAY = can_display()

# ------------------------------------------------------------
#  VERSION 2 : AUCUNE DISTORSION / AUGMENTATION D'IMAGE
# ------------------------------------------------------------
def get_training_data(images, batch_size):
    """
    Version sans distorsion.
    On prend directement les images brutes sans random_transform ni random_warp.
    """
    indices = np.random.randint(len(images), size=batch_size)
    for i, index in enumerate(indices):
        image = images[index]
        warped_img = image.copy()
        target_img = image.copy()

        if i == 0:
            warped_images = np.empty((batch_size,) + warped_img.shape, warped_img.dtype)
            target_images = np.empty((batch_size,) + target_img.shape, target_img.dtype)

        warped_images[i] = warped_img
        target_images[i] = target_img

    return warped_images, target_images


def random_transform(image, *args, **kwargs):
    """Ne fait rien : retourne l'image originale."""
    return image


def random_warp(image):
    """Ne fait rien : retourne simplement l'image originale pour compatibilité."""
    return image.copy(), image.copy()

# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------
class FaceData(Dataset):
    def __init__(self, data_path):
        self.image_files_src = glob(data_path + '/src/aligned/*.jpg')
        self.image_files_dst = glob(data_path + '/dst/aligned/*.jpg')

    def __len__(self):
        return min(len(self.image_files_src), len(self.image_files_dst))

    def __getitem__(self, inx):
        image_file_src = choice(self.image_files_src)
        image_file_dst = choice(self.image_files_dst)
        image_src = np.asarray(Image.open(image_file_src).resize((192, 192))) / 255.
        image_dst = np.asarray(Image.open(image_file_dst).resize((192, 192))) / 255.
        return image_src, image_dst

    def collate_fn(self, batch):
        images_src, images_dst = list(zip(*batch))
        warp_image_src, target_image_src = get_training_data(images_src, len(images_src))
        warp_image_src = torch.tensor(warp_image_src, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        target_image_src = torch.tensor(target_image_src, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        warp_image_dst, target_image_dst = get_training_data(images_dst, len(images_dst))
        warp_image_dst = torch.tensor(warp_image_dst, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        target_image_dst = torch.tensor(target_image_dst, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        return warp_image_src, target_image_src, warp_image_dst, target_image_dst

# ------------------------------------------------------------
# BLOCS DU MODÈLE QUICK96
# ------------------------------------------------------------
def pixel_norm(x, dim=-1):
    return x / torch.sqrt(torch.mean(x ** 2, dim=dim, keepdim=True) + 1e-06)

def depth_to_space(x, size=2):
    b, c, h, w = x.shape
    out_h = size * h
    out_w = size * w
    out_c = c // (size * size)
    x = x.reshape((-1, size, size, out_c, h, w))
    x = x.permute((0, 3, 4, 1, 5, 2))
    x = x.reshape((-1, out_c, out_h, out_w))
    return x

class DepthToSpace(nn.Module):
    def forward(self, x, size=2):
        return depth_to_space(x, size)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 512, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1, True),
            nn.Flatten(),
        )

    def forward(self, x):
        return pixel_norm(self.encoder(x), dim=-1)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding='same'),
            nn.LeakyReLU(0.1, True),
            DepthToSpace()
        )

    def forward(self, x):
        return self.upsample(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding='same')

    def forward(self, x):
        y = nn.functional.leaky_relu(self.conv1(x), 0.2)
        y = self.conv2(y)
        return nn.functional.leaky_relu(y + x, 0.2)

class Inter(nn.Module):
    def __init__(self, input_dim=12800):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1152)
        self.unflatten = nn.Unflatten(1, (128, 3, 3))
        self.upsample = Upsample(128, 512)

    def forward(self, x):
        if x.shape[1] != self.fc1.in_features:
            self.fc1 = nn.Linear(x.shape[1], 128).to(x.device)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.unflatten(x)
        x = self.upsample(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            Upsample(128, 2048),
            ResBlock(512),
            Upsample(512, 1024),
            ResBlock(256),
            Upsample(256, 512),
            ResBlock(128)
        )
        self.conv_outs = nn.ModuleList([
            nn.Conv2d(128, 3, 1, padding='same'),
            nn.Conv2d(128, 3, 3, padding='same'),
            nn.Conv2d(128, 3, 3, padding='same'),
            nn.Conv2d(128, 3, 3, padding='same')
        ])
        self.depth_to_space = DepthToSpace()

    def forward(self, x):
        x = self.decoder(x)
        outs = [conv(x) for conv in self.conv_outs]
        x = torch.concat(outs, 1)
        x = self.depth_to_space(x, 2)
        return torch.sigmoid(x)

# ------------------------------------------------------------
# MÉTRIQUES ET AFFICHAGE
# ------------------------------------------------------------
def create_window(size=11, sigma=1.5, channels=1):
    gk1d = torch.tensor(cv2.getGaussianKernel(size, sigma), dtype=torch.float32)
    gk2d = gk1d @ gk1d.t()
    return gk2d.expand((channels, 1, size, size)).contiguous().clone()

def dssim(image1, image2, window_size=11):
    pad = window_size // 2
    window = create_window(window_size, channels=3).to(device)
    mu1 = nn.functional.conv2d(image1, window, padding=pad, groups=3)
    mu2 = nn.functional.conv2d(image2, window, padding=pad, groups=3)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1*mu2
    sig1_sq = nn.functional.conv2d(image1*image1, window, padding=pad, groups=3) - mu1_sq
    sig2_sq = nn.functional.conv2d(image2*image2, window, padding=pad, groups=3) - mu2_sq
    sig12 = nn.functional.conv2d(image1*image2, window, padding=pad, groups=3) - mu12
    eps = 1e-4
    sig1_sq = torch.clamp(sig1_sq, min=eps)
    sig2_sq = torch.clamp(sig2_sq, min=eps)
    sig12 = torch.clamp(sig12, min=-1.0, max=1.0)
    C1, C2, C3 = 0.01**2, 0.03**2, (0.03**2)/2
    lum = (2*mu12 + C1)/(mu1_sq + mu2_sq + C1)
    con = (2*torch.sqrt(sig1_sq*sig2_sq) + C2)/(sig1_sq + sig2_sq + C2)
    strc = (sig12 + C3)/(torch.sqrt(sig1_sq*sig2_sq) + C3)
    return (1 - (lum*con*strc).mean()) / 2

# ------------------------------------------------------------
# ENTRAÎNEMENT
# ------------------------------------------------------------
def train(data_path: str, model_name='Quick96_no_distortion', new_model=True, saved_models_dir='saved_model'):

    saved_models_dir = Path(saved_models_dir)
    lr = 1e-4
    dataset = FaceData(data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn)

    encoder = Encoder().to(device)
    inter = Inter().to(device)
    decoder_src = Decoder().to(device)
    decoder_dst = Decoder().to(device)

    optim_encoder = torch.optim.Adam(
        [{"params": encoder.parameters()}, {"params": inter.parameters()}], lr=lr)
    optim_decoder_src = torch.optim.Adam(decoder_src.parameters(), lr=lr)
    optim_decoder_dst = torch.optim.Adam(decoder_dst.parameters(), lr=lr)
    criterion_L2 = nn.MSELoss()

    epoch, mean_loss_src, mean_loss_dst = 0, [], []

    print(f"{len(dataloader.dataset)} images, {len(dataloader)} batches.")
    encoder.train(), inter.train(), decoder_src.train(), decoder_dst.train()

    try:
        while True:
            epoch += 1
            mean_epoch_loss_src, mean_epoch_loss_dst = [], []
            for ii, (warp_im_src, target_im_src, warp_im_dst, target_im_dst) in enumerate(
                tqdm(dataloader, desc=f"Epoch {epoch}")
            ):
                # SRC
                latent_src = inter(encoder(warp_im_src))
                reconstruct_im_src = decoder_src(latent_src)
                loss_src_val = dssim(reconstruct_im_src, target_im_src) + criterion_L2(reconstruct_im_src, target_im_src)
                optim_encoder.zero_grad(); optim_decoder_src.zero_grad()
                loss_src_val.backward()
                optim_encoder.step(); optim_decoder_src.step()

                # DST
                latent_dst = inter(encoder(warp_im_dst))
                reconstruct_im_dst = decoder_dst(latent_dst)
                loss_dst_val = dssim(reconstruct_im_dst, target_im_dst) + criterion_L2(reconstruct_im_dst, target_im_dst)
                optim_encoder.zero_grad(); optim_decoder_dst.zero_grad()
                loss_dst_val.backward()
                optim_encoder.step(); optim_decoder_dst.step()

                mean_epoch_loss_src.append(loss_src_val.item())
                mean_epoch_loss_dst.append(loss_dst_val.item())

            mean_loss_src.append(np.mean(mean_epoch_loss_src))
            mean_loss_dst.append(np.mean(mean_epoch_loss_dst))
            print(f"Epoch {epoch}: SRC={mean_loss_src[-1]:.4f}, DST={mean_loss_dst[-1]:.4f}")

            # Sauvegarde périodique
            saved_models_dir.mkdir(exist_ok=True, parents=True)
            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'inter': inter.state_dict(),
                'decoder_src': decoder_src.state_dict(),
                'decoder_dst': decoder_dst.state_dict(),
                'mean_loss_src': mean_loss_src,
                'mean_loss_dst': mean_loss_dst
            }, str(saved_models_dir / f"{model_name}.pth"))

    except KeyboardInterrupt:
        print("\n Entraînement interrompu — modèle sauvegardé.")
