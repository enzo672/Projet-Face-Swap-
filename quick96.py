import sys
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from random import choice
from glob import glob
from pathlib import Path
from tqdm import tqdm

plt.style.use('dark_background')
px = 1 / plt.rcParams['figure.dpi']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------------------------------------------------------------------------
# ----------------------- Data augmentation utils --------------------------
# --------------------------------------------------------------------------
random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.5,
}

def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    h, w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REFLECT)
    if np.random.random() < random_flip:
        result = result[:, ::-1]
    return result

def random_warp(image):
    h, w = image.shape[:2]
    range_ = np.linspace(h / 2 - h * 0.4, h / 2 + h * 0.4, 5)
    mapx = np.broadcast_to(range_, (5, 5)).copy()
    mapy = mapx.T.copy()
    noise = 3 * h / 256
    mapx += np.random.normal(size=(5, 5), scale=noise)
    mapy += np.random.normal(size=(5, 5), scale=noise)
    interp_mapx = cv2.resize(mapx, (w, h)).astype('float32')
    interp_mapy = cv2.resize(mapy, (w, h)).astype('float32')
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    warped_image = np.nan_to_num(np.clip(warped_image, 0, 1))
    return warped_image, image.copy()

def get_training_data(images, batch_size):
    indices = np.random.randint(len(images), size=batch_size)
    for i, index in enumerate(indices):
        image = images[index]
        image = random_transform(image, **random_transform_args)
        warped_img, target_img = random_warp(image)
        if i == 0:
            warped_images = np.empty((batch_size,) + warped_img.shape, warped_img.dtype)
            target_images = np.empty((batch_size,) + target_img.shape, warped_img.dtype)
        warped_images[i] = warped_img
        target_images[i] = target_img
    return warped_images, target_images

# --------------------------------------------------------------------------
# ---------------------------- Dataset class -------------------------------
# --------------------------------------------------------------------------
class FaceData(Dataset):
    def __init__(self, data_path):
        self.image_files_src = [f for f in glob(data_path + '/src/aligned/*.jpg') if cv2.haveImageReader(f)]
        self.image_files_dst = [f for f in glob(data_path + '/dst/aligned/*.jpg') if cv2.haveImageReader(f)]

    def __len__(self):
        return min(len(self.image_files_src), len(self.image_files_dst))

    def __getitem__(self, inx):
        image_file_src = choice(self.image_files_src)
        image_file_dst = choice(self.image_files_dst)
        image_src = np.asarray(Image.open(image_file_src).convert('RGB').resize((192, 192)), dtype=np.float32) / 255.
        image_dst = np.asarray(Image.open(image_file_dst).convert('RGB').resize((192, 192)), dtype=np.float32) / 255.
        image_src = np.nan_to_num(np.clip(image_src, 0, 1))
        image_dst = np.nan_to_num(np.clip(image_dst, 0, 1))
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

# --------------------------------------------------------------------------
# ----------------------------- Model blocks -------------------------------
# --------------------------------------------------------------------------
def pixel_norm(x, dim=-1, eps=1e-8):
    norm = torch.sqrt(torch.mean(x ** 2, dim=dim, keepdim=True) + eps)
    norm = torch.clamp(norm, min=eps)
    return x / norm

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
        x = self.encoder(x)
        return pixel_norm(x, dim=-1)

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
    def __init__(self, input_dim=18432):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1152)
        self.unflatten = nn.Unflatten(1, (128, 3, 3))
        self.upsample = Upsample(128, 512)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.unflatten(x)
        x = self.upsample(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # S'assure que le nombre de canaux de sortie soit multiple de 4
        out_channels = ((out_channels + 3) // 4) * 4
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.shuffle = DepthToSpace()

    def forward(self, x):
        x = self.act(self.conv(x))
        return self.shuffle(x, 2)



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
            nn.Conv2d(128, 3, 3, padding='same')
            for _ in range(4)
        ])
        self.depth_to_space = DepthToSpace()

    def forward(self, x):
        x = self.decoder(x)
        x = sum(conv(x) for conv in self.conv_outs) / 4
        x = self.depth_to_space(x, 2)
        return torch.sigmoid(x)

# --------------------------------------------------------------------------
# ------------------------- DSSIM STABILISÉ -------------------------------
# --------------------------------------------------------------------------
def dssim(image1, image2):
    return torch.mean((image1 - image2) ** 2)  # remplace SSIM instable

# --------------------------------------------------------------------------
# ------------------------------- TRAIN ------------------------------------
# --------------------------------------------------------------------------
def train(data_path, model_name='Quick96', new_model=False, saved_models_dir='saved_model'):
    saved_models_dir = Path(saved_models_dir)
    lr = 5e-5
    dataset = FaceData(data_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)

    encoder, inter, dec_src, dec_dst = Encoder().to(device), Inter().to(device), Decoder().to(device), Decoder().to(device)
    opt_e = torch.optim.Adam(list(encoder.parameters()) + list(inter.parameters()), lr=lr)
    opt_s = torch.optim.Adam(dec_src.parameters(), lr=lr)
    opt_d = torch.optim.Adam(dec_dst.parameters(), lr=lr)
    mse = nn.MSELoss()

    epoch, l_src, l_dst = 0, [], []

    print(f"{len(dataset)} images, {len(dataloader)} batches.")
    try:
        while True:
            epoch += 1
            e_src, e_dst = [], []
            for ws, ts, wd, td in tqdm(dataloader, desc=f"Epoch {epoch}"):
                r_s = torch.clamp(dec_src(inter(encoder(ws))), 0, 1)
                loss_s = 0.8*mse(r_s, ts) + 0.2*dssim(r_s, ts)
                opt_e.zero_grad(); opt_s.zero_grad(); loss_s.backward(); opt_e.step(); opt_s.step()

                r_d = torch.clamp(dec_dst(inter(encoder(wd))), 0, 1)
                loss_d = 0.8*mse(r_d, td) + 0.2*dssim(r_d, td)
                opt_e.zero_grad(); opt_d.zero_grad(); loss_d.backward(); opt_e.step(); opt_d.step()

                e_src.append(loss_s.item()); e_dst.append(loss_d.item())

            l_src.append(np.mean(e_src)); l_dst.append(np.mean(e_dst))
            print(f"Epoch {epoch} | Loss_src={l_src[-1]:.4f} | Loss_dst={l_dst[-1]:.4f}")

    except KeyboardInterrupt:
        print("Sauvegarde avant sortie...")
        torch.save({
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'inter': inter.state_dict(),
            'decoder_src': dec_src.state_dict(),
            'decoder_dst': dec_dst.state_dict(),
        }, saved_models_dir / f"{model_name}.pth")
        print(f"Modèle sauvegardé : {saved_models_dir / f'{model_name}.pth'}")
