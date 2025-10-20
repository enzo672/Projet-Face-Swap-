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

torch.autograd.set_detect_anomaly(True)
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
    b, c, h, w = x.size()
    oc = c // (size ** 2)
    x = x.view(b, size, size, oc, h, w)
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
    x = x.view(b, oc, h * size, w * size)
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
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.1, True),
            DepthToSpace()
        )
    def forward(self, x):
        return self.upsample(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
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
            nn.Conv2d(128, 3, 1, padding=1),
            nn.Conv2d(128, 3, 3, padding=1),
            nn.Conv2d(128, 3, 3, padding=1),
            nn.Conv2d(128, 3, 3, padding=1)
        ])
        self.depth_to_space = DepthToSpace()

    def forward(self, x):
        x = self.decoder(x)
        outs = [conv(x) for conv in self.conv_outs]
        x = torch.cat(outs, dim=1)
        x = self.depth_to_space(x, 2)
        return torch.sigmoid(x)


# --------------------------------------------------------------------------
# ------------------------------- DSSIM ------------------------------------
# --------------------------------------------------------------------------
def create_window(size=11, sigma=1.5, channels=1):
    gk1d = torch.tensor(cv2.getGaussianKernel(size, sigma), dtype=torch.float32)
    gk2d = gk1d @ gk1d.t()
    return gk2d.expand((channels, 1, size, size)).contiguous().clone()

def dssim(image1, image2, window_size=11, eps=1e-6):
    pad = window_size // 2
    window = create_window(window_size, channels=3).to(image1.device)
    mu1 = nn.functional.conv2d(image1, window, padding=pad, groups=3)
    mu2 = nn.functional.conv2d(image2, window, padding=pad, groups=3)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2
    sig1_sq = torch.clamp(nn.functional.conv2d(image1 * image1, window, padding=pad, groups=3) - mu1_sq, min=eps)
    sig2_sq = torch.clamp(nn.functional.conv2d(image2 * image2, window, padding=pad, groups=3) - mu2_sq, min=eps)
    sig12 = nn.functional.conv2d(image1 * image2, window, padding=pad, groups=3) - mu12
    C1, C2, C3 = 0.01**2, 0.03**2, (0.03**2) / 2
    lum = (2 * mu12 + C1) / (mu1_sq + mu2_sq + C1)
    con = (2 * torch.sqrt(sig1_sq * sig2_sq) + C2) / (sig1_sq + sig2_sq + C2)
    strc = (sig12 + C3) / (torch.sqrt(sig1_sq * sig2_sq) + C3)
    dssim_map = (1 - (lum * con * strc)) / 2
    return torch.mean(dssim_map)


# --------------------------------------------------------------------------
# ----------------------------- Visualization ------------------------------
# --------------------------------------------------------------------------
def draw_results(reconstruct_src, target_src, reconstruct_dst, target_dst, fake, loss_src, loss_dst):
    fig, axes = plt.subplots(figsize=(660 * px, 370 * px))
    axes.plot(loss_src, label='loss src')
    axes.plot(loss_dst, label='loss dst')
    plt.legend()
    plt.title(f'press q to quit and save, or r to refresh\nepoch = {len(loss_src)}')
    canvas = fig.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3)) / 255.

    images_for_grid = []
    for ii in range(min(3, len(reconstruct_src))):
        images_for_grid.extend([
            reconstruct_src[ii],
            target_src[ii],
            reconstruct_dst[ii],
            target_dst[ii],
            fake[ii]
        ])

    im_grid = torchvision.utils.make_grid([img.cpu() for img in images_for_grid], nrow=5, padding=30)
    im_grid = im_grid.permute(1, 2, 0).numpy()
    final_image = np.vstack([image_array, im_grid])
    final_image = final_image[..., ::-1]  # RGB -> BGR for OpenCV
    return final_image


# --------------------------------------------------------------------------
# ------------------------------ Training loop -----------------------------
# --------------------------------------------------------------------------
def train(data_path: str, model_name='Quick96', new_model=False, saved_models_dir='saved_model'):
    saved_models_dir = Path(saved_models_dir)
    lr = 5e-5
    dataset = FaceData(data_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)

    encoder = Encoder().to(device)
    inter = Inter().to(device)
    decoder_src = Decoder().to(device)
    decoder_dst = Decoder().to(device)

    optim_encoder = torch.optim.Adam(
        [{"params": encoder.parameters()}, {"params": inter.parameters()}], lr=lr)
    optim_decoder_src = torch.optim.Adam(decoder_src.parameters(), lr=lr)
    optim_decoder_dst = torch.optim.Adam(decoder_dst.parameters(), lr=lr)
    criterion_L2 = nn.MSELoss()

    saved_model, epoch, mean_loss_src, mean_loss_dst = {}, 0, [], []
    model_path = saved_models_dir / f'{model_name}.pth'

    if not new_model and model_path.exists():
        print(f'[info] Loading previous model from {model_path}')
        saved_model = torch.load(str(model_path), map_location=device)
        epoch = saved_model['epoch']
        encoder.load_state_dict(saved_model['encoder'])
        inter.load_state_dict(saved_model['inter'])
        decoder_src.load_state_dict(saved_model['decoder_src'])
        decoder_dst.load_state_dict(saved_model['decoder_dst'])
        optim_encoder.load_state_dict(saved_model['optimizer_encoder'])
        optim_decoder_src.load_state_dict(saved_model['optimizer_decoder_src'])
        optim_decoder_dst.load_state_dict(saved_model['optimizer_decoder_dst'])
        mean_loss_src = saved_model.get('mean_loss_src', [])
        mean_loss_dst = saved_model.get('mean_loss_dst', [])

    encoder.train(); inter.train(); decoder_src.train(); decoder_dst.train()
    first_run, run = True, True
    print(f"[info] {len(dataloader.dataset)} images, {len(dataloader)} batches.")

    while run:
        epoch += 1
        epoch_loss_src, epoch_loss_dst = [], []

        for ii, (warp_im_src, target_im_src, warp_im_dst, target_im_dst) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch}")
        ):
            # SOURCE
            latent_src = inter(encoder(warp_im_src))
            reconstruct_im_src = decoder_src(latent_src)
            loss_src_val = dssim(reconstruct_im_src, target_im_src) + criterion_L2(reconstruct_im_src, target_im_src)
            optim_encoder.zero_grad(); optim_decoder_src.zero_grad()
            loss_src_val.backward(retain_graph=True)
            optim_encoder.step(); optim_decoder_src.step()

            # DESTINATION
            latent_dst = inter(encoder(warp_im_dst))
            reconstruct_im_dst = decoder_dst(latent_dst)
            loss_dst_val = dssim(reconstruct_im_dst, target_im_dst) + criterion_L2(reconstruct_im_dst, target_im_dst)
            optim_encoder.zero_grad(); optim_decoder_dst.zero_grad()
            loss_dst_val.backward()
            optim_encoder.step(); optim_decoder_dst.step()

            epoch_loss_src.append(loss_src_val.item())
            epoch_loss_dst.append(loss_dst_val.item())

            # Visualisation à la première itération
            if first_run:
                first_run = False
                plt.ioff()
                fake = decoder_src(inter(encoder(target_im_dst)))
                result_image = draw_results(
                    reconstruct_im_src, target_im_src,
                    reconstruct_im_dst, target_im_dst,
                    fake, mean_loss_src, mean_loss_dst
                )
                cv2.imshow('results', result_image)
                cv2.waitKey(1)

            k = cv2.waitKey(1)
            if k == ord('q'):
                run = False
                break
            elif k == ord('r'):
                fake = decoder_src(inter(encoder(target_im_dst)))
                result_image = draw_results(
                    reconstruct_im_src, target_im_src,
                    reconstruct_im_dst, target_im_dst,
                    fake, mean_loss_src, mean_loss_dst
                )
                cv2.imshow('results', result_image)
                cv2.waitKey(1)

        mean_loss_src.append(np.mean(epoch_loss_src))
        mean_loss_dst.append(np.mean(epoch_loss_dst))

        # Sauvegarde auto toutes les 5 époques
        if epoch % 5 == 0:
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
                'mean_loss_dst': mean_loss_dst,
            }
            saved_models_dir.mkdir(exist_ok=True, parents=True)
            torch.save(saved_model, str(model_path))
            print(f"[saved] Model saved at epoch {epoch}")

    cv2.destroyAllWindows()
