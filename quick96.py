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

def can_display():
    if platform.system() == "Linux":
        return "DISPLAY" in os.environ and bool(os.environ["DISPLAY"])
    elif platform.system() in ["Darwin", "Windows"]:
        return True
    return False

HAS_DISPLAY = can_display()

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

    # clamp toutes les variances pour éviter 0 ou NaN
    eps = 1e-4
    sig1_sq = torch.clamp(sig1_sq, min=eps)
    sig2_sq = torch.clamp(sig2_sq, min=eps)
    sig12 = torch.clamp(sig12, min=-1.0, max=1.0)  # pas trop extrême

    C1, C2, C3 = 0.01**2, 0.03**2, (0.03**2)/2
    lum = (2*mu12 + C1)/(mu1_sq + mu2_sq + C1)
    con = (2*torch.sqrt(sig1_sq*sig2_sq) + C2)/(sig1_sq + sig2_sq + C2)
    strc = (sig12 + C3)/(torch.sqrt(sig1_sq*sig2_sq) + C3)
    return (1 - (lum*con*strc).mean()) / 2


def draw_results(reconstruct_src, target_src, reconstruct_dst, target_dst, fake, loss_src, loss_dst):
    # Clamp + nan_to_num pour toutes les images
    reconstruct_src = torch.clamp(torch.nan_to_num(reconstruct_src, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    reconstruct_dst = torch.clamp(torch.nan_to_num(reconstruct_dst, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    target_src = torch.clamp(torch.nan_to_num(target_src, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    target_dst = torch.clamp(torch.nan_to_num(target_dst, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    fake = torch.clamp(torch.nan_to_num(fake, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    # Convertir en numpy
    def safe_to_numpy(tensor):
        arr = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        return arr

    reconstruct_src = safe_to_numpy(reconstruct_src)
    reconstruct_dst = safe_to_numpy(reconstruct_dst)
    target_src = safe_to_numpy(target_src)
    target_dst = safe_to_numpy(target_dst)
    fake = safe_to_numpy(fake)

    # Graphique des losses
    dpi = plt.rcParams['figure.dpi']
    fig, axes = plt.subplots(figsize=(660 / dpi, 370 / dpi))
    axes.plot(loss_src, label='loss src')
    axes.plot(loss_dst, label='loss dst')
    axes.legend()
    axes.set_title(f'press q to quit and save, or r to refresh\nEpoch = {len(loss_src)}')
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plt.close(fig)

    image_array = buffer.reshape((height, width, 3)) / 255.0

    # Grille d’images
    images_for_grid = []
    for ii in range(min(3, len(reconstruct_src))):
        images_for_grid.extend([
            reconstruct_src[ii],
            target_src[ii],
            reconstruct_dst[ii],
            target_dst[ii],
            fake[ii]
        ])

    im_grid = torchvision.utils.make_grid(
        [torch.tensor(img).permute(2,0,1) for img in images_for_grid],
        nrow=5, padding=30
    ).permute(1, 2, 0).cpu().numpy()

    target_w = min(image_array.shape[1], im_grid.shape[1])
    image_array = cv2.resize(image_array, (target_w, image_array.shape[0]))
    im_grid = cv2.resize(im_grid, (target_w, im_grid.shape[0]))

    final_image = np.vstack([image_array, im_grid])
    final_image = np.clip(final_image[..., ::-1] * 255, 0, 255).astype(np.uint8)

    return final_image



# entrainement
def train(data_path: str, model_name='Quick96', new_model=False, saved_models_dir='saved_model'):

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

    # Charger ancien modèle si existant
    if not new_model and (saved_models_dir / f'{model_name}.pth').exists():
        print('Loading previous model...')
        saved_model = torch.load(str(saved_models_dir / f'{model_name}.pth'))
        epoch = saved_model['epoch']
    else:
        saved_model, epoch = {}, 0
        mean_loss_src, mean_loss_dst = [], []

    if saved_model:
        print('loading model states')
        encoder.load_state_dict(saved_model['encoder'])
        inter.load_state_dict(saved_model['inter'])
        decoder_src.load_state_dict(saved_model['decoder_src'])
        decoder_dst.load_state_dict(saved_model['decoder_dst'])
        optim_encoder.load_state_dict(saved_model['optimizer_encoder'])
        optim_decoder_src.load_state_dict(saved_model['optimizer_decoder_src'])
        optim_decoder_dst.load_state_dict(saved_model['optimizer_decoder_dst'])
        mean_loss_src = saved_model['mean_loss_src']
        mean_loss_dst = saved_model['mean_loss_dst']

    encoder.train(), inter.train(), decoder_src.train(), decoder_dst.train()
    torch.autograd.set_detect_anomaly(True)

    print(f"{len(dataloader.dataset)} images, {len(dataloader)} batches.")
    first_run, run = True, True

    try:
        while run:
            epoch += 1
            mean_epoch_loss_src, mean_epoch_loss_dst = [], []

            for ii, (warp_im_src, target_im_src, warp_im_dst, target_im_dst) in enumerate(
                tqdm(dataloader, desc=f"Epoch {epoch}")
            ):
                # SRC 
                latent_src = inter(encoder(warp_im_src))
                reconstruct_im_src = decoder_src(latent_src)
                reconstruct_im_src = torch.clamp(torch.nan_to_num(reconstruct_im_src, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

                loss_src_val = dssim(reconstruct_im_src, target_im_src) + criterion_L2(reconstruct_im_src, target_im_src)
                optim_encoder.zero_grad()
                optim_decoder_src.zero_grad()
                loss_src_val.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(inter.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(decoder_src.parameters(), max_norm=1.0)
                optim_encoder.step()
                optim_decoder_src.step()

                # DST 
                latent_dst = inter(encoder(warp_im_dst))
                reconstruct_im_dst = decoder_dst(latent_dst)
                reconstruct_im_dst = torch.clamp(torch.nan_to_num(reconstruct_im_dst, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

                loss_dst_val = dssim(reconstruct_im_dst, target_im_dst) + criterion_L2(reconstruct_im_dst, target_im_dst)
                optim_encoder.zero_grad()
                optim_decoder_dst.zero_grad()
                loss_dst_val.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(inter.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(decoder_dst.parameters(), max_norm=1.0)
                optim_encoder.step()
                optim_decoder_dst.step()

                mean_epoch_loss_src.append(loss_src_val.item())
                mean_epoch_loss_dst.append(loss_dst_val.item())

                # Affichage 
                if first_run:
                    first_run = False
                    plt.ioff()
                    fake = decoder_src(inter(encoder(target_im_dst)))
                    fake = torch.clamp(torch.nan_to_num(fake, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

                    result_image = draw_results(
                        reconstruct_im_src, target_im_src, reconstruct_im_dst, target_im_dst, fake,
                        mean_loss_src, mean_loss_dst
                    )
                    if HAS_DISPLAY:
                        cv2.imshow('results', result_image)
                        cv2.waitKey(1)
                    else:
                        Path("saved_results").mkdir(exist_ok=True)
                        cv2.imwrite(f"saved_results/epoch_{epoch:04d}.jpg", result_image)

                k = cv2.waitKey(1)
                if k == ord('q'):
                    run = False
                    break
                elif k == ord('r'):
                    # recalcul et clamp du fake
                    fake = decoder_src(inter(encoder(target_im_dst)))
                    fake = torch.clamp(torch.nan_to_num(fake, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
                    stable_reconstruct_src = torch.clamp(torch.nan_to_num(reconstruct_im_src, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
                    stable_reconstruct_dst = torch.clamp(torch.nan_to_num(reconstruct_im_dst, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

                    result_image = draw_results(
                        stable_reconstruct_src, target_im_src, stable_reconstruct_dst, target_im_dst, fake,
                        mean_loss_src, mean_loss_dst
                    )

                    if HAS_DISPLAY:
                        cv2.imshow('results', result_image)
                        cv2.waitKey(1)
                    else:
                        Path("saved_results").mkdir(exist_ok=True)
                        cv2.imwrite(f"saved_results/epoch_{epoch:04d}_refresh.jpg", result_image)

            mean_loss_src.append(np.mean(mean_epoch_loss_src))
            mean_loss_dst.append(np.mean(mean_epoch_loss_dst))

    except KeyboardInterrupt:
        print("\n[info] Interruption détectée — sauvegarde du modèle...")
        saved_model = {
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'inter': inter.state_dict(),
            'decoder_src': decoder_src.state_dict(),
            'decoder_dst': decoder_dst.state_dict(),
            'optimizer_encoder': optim_encoder.state_dict(),
            'optimizer_decoder_src': optim_decoder_src.state_dict(),
            'optimizer_decoder_dst': optim_decoder_dst.state_dict(),
            'mean_loss_src': mean_loss_src,
            'mean_loss_dst': mean_loss_dst
        }
        saved_models_dir.mkdir(exist_ok=True, parents=True)
        torch.save(saved_model, str(saved_models_dir / f"{model_name}.pth"))
        print(f"[saved] Modèle sauvegardé → {saved_models_dir / (model_name + '.pth')}")

    finally:
        if HAS_DISPLAY:
            cv2.destroyAllWindows()
