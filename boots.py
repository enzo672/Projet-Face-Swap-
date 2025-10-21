# bootstrap_colorization.py
import os, textwrap, pathlib

ROOT = pathlib.Path.home() / "colorization"
SRC = ROOT / "src"
(VIN := ROOT / "vintage").mkdir(parents=True, exist_ok=True)
(ROOT / "runs").mkdir(parents=True, exist_ok=True)
SRC.mkdir(parents=True, exist_ok=True)

def write(path, content):
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    print("Wrote", path.relative_to(ROOT))

# requirements (optionnel)
write(ROOT / "requirements.txt", """
torch
torchvision
torchaudio
torcheval
opencv-python
pillow
matplotlib
scikit-image
tqdm
scikit-learn
""")

# unet.py
write(SRC / "unet.py", """
import torch
import torch.nn as nn

def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, base=64):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.enc2 = conv_block(base, base*2)
        self.enc3 = conv_block(base*2, base*4)
        self.enc4 = conv_block(base*4, base*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base*8, base*16)
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = conv_block(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = conv_block(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = conv_block(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = conv_block(base*2, base)
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.up4(b); d4 = torch.cat([d4, e4], dim=1); d4 = self.dec4(d4)
        d3 = self.up3(d4); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        return self.head(d1)
""")

# vintage_augs.py
write(SRC / "vintage_augs.py", """
import torch
import torchvision.transforms.functional as TF
import random, math
from PIL import Image
import numpy as np

class MakeVintage:
    \"\"\"Transforme une image RGB en 'photo d'époque' (entrée du réseau).\"\"\"
    def __init__(self, out_mode=\"L\"):
        self.out_mode = out_mode

    def __call__(self, img: Image.Image):
        img = TF.gaussian_blur(img, kernel_size=3)
        gray = TF.rgb_to_grayscale(img, num_output_channels=1)
        if self.out_mode == \"RGB\":
            arr = np.array(img).astype(np.float32)
            M = np.array([[0.393, 0.769, 0.189],
                          [0.349, 0.686, 0.168],
                          [0.272, 0.534, 0.131]], dtype=np.float32)
            sepia = np.clip(arr @ M.T, 0, 255).astype(np.uint8)
            t = TF.to_tensor(Image.fromarray(sepia))
        else:
            t = TF.to_tensor(gray)

        noise = torch.randn_like(t) * 0.03
        t = torch.clamp(t + noise, 0., 1.)

        if random.random() < 0.3:
            c, h, w = t.shape
            for _ in range(random.randint(1,3)):
                x0 = random.randint(0, w-1)
                for y in range(h):
                    x = min(w-1, max(0, int(x0 + 2*math.sin(y/10.0))))
                    t[:, y, x:x+1] = 1.0
        return t
""")

# dataset.py
write(SRC / "dataset.py", """
from torchvision.datasets import ImageNet
from torchvision import transforms
from torch.utils.data import Subset, DataLoader, Dataset
import random
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
from typing import Sequence, Optional
from vintage_augs import MakeVintage

class PairDataset(Dataset):
    \"\"\"Retourne (x_vintage, y_rgb) à partir d'un dataset RGB.\"\"\"
    def __init__(self, base_ds, size=224, out_mode=\"L\"):
        self.base = base_ds
        self.base_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size)])
        self.to_tensor = transforms.ToTensor()
        self.vintage = MakeVintage(out_mode=out_mode)

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        img, y = self.base[i]
        img = self.base_tf(img.convert(\"RGB\"))
        target_rgb = self.to_tensor(img)
        x_vintage  = self.vintage(img)
        return x_vintage, target_rgb

def pick_classes(n_classes, seed=42):
    random.seed(seed)
    return sorted(random.sample(range(1000), n_classes))

def _labels_of(ds):
    if isinstance(ds, Subset):
        return [ds.dataset[i][1] for i in ds.indices]
    else:
        return [ds[i][1] for i in range(len(ds))]

def build_imagenet_loaders(root: str,
                           classes_to_keep: Optional[Sequence[int]]=None,
                           batch_size: int = 16,
                           num_workers: int = 8,
                           seed: int = 42,
                           out_mode: str = \"L\"):
    base_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    train_full = ImageNet(root=root, split=\"train\", transform=base_tf)
    val_off    = ImageNet(root=root, split=\"val\",   transform=base_tf)

    if classes_to_keep is not None:
        cls = set(classes_to_keep)
        keep_tr = [i for i,(_,y) in enumerate(train_full) if y in cls]
        keep_va = [i for i,(_,y) in enumerate(val_off)    if y in cls]
        train_full = Subset(train_full, keep_tr)
        val_off    = Subset(val_off, keep_va)

    ys = _labels_of(train_full)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    idx_tr, idx_va = next(sss.split(range(len(ys)), ys))

    if isinstance(train_full, Subset):
        base_idx = train_full.indices
        idx_tr = [base_idx[i] for i in idx_tr]
        idx_va = [base_idx[i] for i in idx_va]
        train_rgb = Subset(train_full.dataset, idx_tr)
        val_rgb   = Subset(train_full.dataset, idx_va)
    else:
        train_rgb = Subset(train_full, idx_tr)
        val_rgb   = Subset(train_full, idx_va)

    train_ds = PairDataset(train_rgb, out_mode=out_mode)
    val_ds   = PairDataset(val_rgb,   out_mode=out_mode)
    test_ds  = PairDataset(val_off,   out_mode=out_mode)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
""")

# train.py
write(SRC / "train.py", """
import os, argparse, torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torcheval.metrics import StructuralSimilarity
from dataset import build_imagenet_loaders, pick_classes
from unet import UNet

def train_one_epoch(model, loader, opt, device):
    model.train()
    crit = nn.L1Loss()
    ssim_metric = StructuralSimilarity(data_range=1.0).to(device)
    tot_loss = tot_ssim = n = 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        if x.ndim == 3: x = x.unsqueeze(1)
        opt.zero_grad()
        yhat = model(x).clamp(0,1)
        loss = crit(yhat, y)
        loss.backward(); opt.step()
        with torch.no_grad():
            ssim_metric.update(yhat, y)
            ssim = ssim_metric.compute().item()
            ssim_metric.reset()
        bs = y.size(0)
        tot_loss += loss.item() * bs
        tot_ssim += ssim * bs
        n += bs
    return tot_loss/n, tot_ssim/n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.L1Loss()
    ssim_metric = StructuralSimilarity(data_range=1.0).to(device)
    tot_loss = n = 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        if x.ndim == 3: x = x.unsqueeze(1)
        yhat = model(x).clamp(0,1)
        l = crit(yhat, y)
        bs = y.size(0)
        tot_loss += l.item() * bs
        n += bs
        ssim_metric.update(yhat, y)
    return tot_loss/n, ssim_metric.compute().item()

def run_experiment(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    classes = pick_classes(args.n_classes, seed=args.seed) if args.n_classes < 1000 else None
    tr, va, te = build_imagenet_loaders(args.data_root, classes_to_keep=classes,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        seed=args.seed, out_mode="L")
    model = UNet(in_ch=1, out_ch=3, base=args.base).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs+1):
        tr_loss, tr_ssim = train_one_epoch(model, tr, opt, device)
        va_loss, va_ssim = evaluate(model, va, device)
        print(f"[{args.n_classes} cls] Epoch {ep:02d} | train L1={tr_loss:.4f} SSIM={tr_ssim:.4f} | "
              f"val L1={va_loss:.4f} SSIM={va_ssim:.4f}")

    te_loss, te_ssim = evaluate(model, te, device)
    print(f"[{args.n_classes} cls] TEST L1={te_loss:.4f} SSIM={te_ssim:.4f}")
    os.makedirs("runs", exist_ok=True)
    ckpt = f"runs/unet_{args.n_classes}cls.pt"
    torch.save(model.state_dict(), ckpt)
    print("Saved:", ckpt)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/imagenet")
    p.add_argument("--n-classes", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run_experiment(args)
""")

# eval.py
write(SRC / "eval.py", """
import argparse, torch
from PIL import Image
from torchvision import transforms
from vintage_augs import MakeVintage
from unet import UNet

@torch.no_grad()
def colorize_image(img_path, ckpt, size=224, device="cuda"):
    device = device if (device=="cpu" or torch.cuda.is_available()) else "cpu"
    model = UNet(in_ch=1, out_ch=3).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    img = Image.open(img_path).convert("RGB")
    base = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size)])
    x_vint = MakeVintage(out_mode="L")(base(img)).unsqueeze(0).to(device)
    y_hat = model(x_vint).clamp(0,1).squeeze(0).cpu()
    out = transforms.ToPILImage()(y_hat)
    out_path = img_path.rsplit(".",1)
    out_path = out_path[0] + "_colorized." + out_path[1]
    out.save(out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    colorize_image(args.img, args.ckpt, device="cpu" if args.cpu else "cuda")
""")

# smoke_test.py (marche sans dataset)
write(SRC / "smoke_test.py", """
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from pathlib import Path
from unet import UNet
from vintage_augs import MakeVintage

def load_or_make_demo():
    vit = Path.cwd().parent / "vintage"
    imgs = list(vit.glob("*.jpg")) + list(vit.glob("*.png")) + list(vit.glob("*.jpeg"))
    if imgs:
        return Image.open(imgs[0]).convert("RGB")
    h, w = 256, 256
    arr = np.zeros((h,w,3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            arr[y,x] = (x%256, y%256, (x+y)%256)
    return Image.fromarray(arr)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, "| Torch:", torch.__version__)
    img = load_or_make_demo()
    base = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    x = MakeVintage(out_mode="L")(base(img)).unsqueeze(0).to(device)
    model = UNet(in_ch=1, out_ch=3, base=32).to(device)
    with torch.no_grad():
        yhat = model(x).cpu().squeeze(0)
    out = transforms.ToPILImage()(yhat.clamp(0,1))
    Path("runs").mkdir(exist_ok=True, parents=True)
    out.save("runs/smoke_out.jpg")
    print("OK -> runs/smoke_out.jpg")

if __name__ == "__main__":
    main()
""")

print(f"\n✅ Projet prêt dans: {ROOT}\n"
      f"- Smoke test:    python src/smoke_test.py\n"
      f"- Entraînement:  python src/train.py --data-root data/imagenet --n-classes 10 --epochs 5\n"
      f"- Inférence:     python src/eval.py --img vintage/ta_photo.jpg --ckpt runs/unet_10cls.pt\n")
