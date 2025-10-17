import sys
import os
import platform
import datetime
import time
from random import choice
from glob import glob
from pathlib import Path
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

# --------------------------------------------------------------------------
# -------- Détection de la capacité d'affichage (Mac / Linux / Windows) -----
# --------------------------------------------------------------------------
def can_display():
    """Vérifie si on peut afficher avec cv2.imshow"""
    if platform.system() == "Linux":
        return "DISPLAY" in os.environ and bool(os.environ["DISPLAY"])
    elif platform.system() in ["Darwin", "Windows"]:
        return True
    return False

HAS_DISPLAY = can_display()

# --------------------------------------------------------------------------
# --------------------------- TRANSFORMATIONS -------------------------------
# --------------------------------------------------------------------------
random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.5,
}

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


def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    h, w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < random_flip:
        result = result[:, ::-1]
    return result


def random_warp(image):
    h, w = image.shape[:2]
    range_ = np.linspace(h / 2 - h * 0.4, h / 2 + h * 0.4, 5)
    mapx = np.broadcast_to(range_, (5, 5))
    mapy = mapx.T
    mapx = mapx + np.random.normal(size=(5, 5), scale=5*h/256)
    mapy = mapy + np.random.normal(size=(5, 5), scale=5*h/256)
    interp_mapx = cv2.resize(mapx, (int(w / 2 * 1.25), int(h / 2 * 1.25)))[int(w/16):int(w*7/16), int(w/16):int(w*7/16)].astype('float32')
    interp_mapy = cv2.resize(mapy, (int(w / 2 * 1.25), int(h / 2 * 1.25)))[int(w/16):int(w*7/16), int(w/16):int(w*7/16)].astype('float32')
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = np.mgrid[0:w//2+1:w//8, 0:h//2+1:h//8].T.reshape(-1, 2)
    A = np.zeros((2 * src_points.shape[0], 2))
    A[0::2, :] = src_points
    A[0::2, 1] = -A[0::2, 1]
    A[1::2, :] = src_points[:, ::-1]
    A = np.hstack((A, np.tile(np.eye(2), (src_points.shape[0], 1))))
    b = dst_points.flatten()
    similarity_mat = np.linalg.lstsq(A, b, rcond=None)[0]
    similarity_mat = np.array([[similarity_mat[0], -similarity_mat[1], similarity_mat[2]],
                               [similarity_mat[1], similarity_mat[0], similarity_mat[3]]])
    target_image = cv2.warpAffine(image, similarity_mat, (w // 2, h // 2))
    return warped_image, target_image

# --------------------------------------------------------------------------
# ----------------------------- DATASET ------------------------------------
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# ----------------------------- RÉSEAU -------------------------------------
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# ------------------------- FONCTION D’AFFICHAGE ---------------------------
# --------------------------------------------------------------------------
def draw_results(reconstruct_src, target_src, reconstruct_dst, target_dst, fake, loss_src, loss_dst):
    dpi = plt.rcParams['figure.dpi']
    fig, axes = plt.subplots(figsize=(660 / dpi, 370 / dpi))
    axes.plot(loss_src, label='loss src')
    axes.plot(loss_dst, label='loss dst')
    axes.legend()
    axes.set_title(f'press q to quit and save, or r to refresh\nEpoch = {len(loss_src)}')

    canvas = fig.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    buffer = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image_array = buffer.reshape((height, width, 3)) / 255.0
    plt.close(fig)

    images_for_grid = []
    for ii in range(min(3, len(reconstruct_src))):
        images_for_grid.extend([
            reconstruct_src[ii],
            target_src[ii],
            reconstruct_dst[ii],
            target_dst[ii],
            fake[ii]
        ])
    im_grid = torchvision.utils.make_grid(images_for_grid, nrow=5, padding=30)
    im_grid = im_grid.permute(1, 2, 0).cpu().numpy()
    final_image = np.vstack([image_array, im_grid])
    final_image = np.clip(final_image[..., ::-1] * 255, 0, 255).astype(np.uint8)
    return final_image

# --------------------------------------------------------------------------
# ---------------------- AFFICHAGE CONDITIONNEL ----------------------------
# --------------------------------------------------------------------------
def show_or_save(result_image, epoch):
    """Affiche ou sauvegarde l'image selon le système"""
    if HAS_DISPLAY:
        cv2.imshow('results', result_image)
        cv2.waitKey(1)
    else:
        Path("saved_results").mkdir(exist_ok=True)
        cv2.imwrite(f"saved_results/epoch_{epoch:04d}.jpg", result_image)
