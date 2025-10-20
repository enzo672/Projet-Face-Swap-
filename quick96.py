import os
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
import platform


def can_display():
    if platform.system() == "Linux":
        return "DISPLAY" in os.environ and bool(os.environ["DISPLAY"])
    return True

HAS_DISPLAY = can_display()
plt.style.use('dark_background')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.5,
}

def get_training_data(images, batch_size):
    indices = np.random.randint(len(images), size=batch_size)
    warped_images, target_images = [], []
    for idx in indices:
        img = random_transform(images[idx], **random_transform_args)
        w, t = random_warp(img)
        warped_images.append(w)
        target_images.append(t)
    return np.array(warped_images), np.array(target_images)

def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    h, w = image.shape[:2]
    rot = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w//2, h//2), rot, scale)
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REFLECT)
    if np.random.rand() < random_flip:
        result = result[:, ::-1]
    return result

def random_warp(image):
    h, w = image.shape[:2]
    range_ = np.linspace(h/2 - h*0.4, h/2 + h*0.4, 5)
    mapx = np.broadcast_to(range_, (5,5))
    mapy = mapx.T
    mapx += np.random.normal(size=(5,5), scale=2*h/256)
    mapy += np.random.normal(size=(5,5), scale=2*h/256)
    interp_mapx = cv2.resize(mapx, (w//2, h//2)).astype('float32')
    interp_mapy = cv2.resize(mapy, (w//2, h//2)).astype('float32')
    warped = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    target = cv2.resize(image, (w//2, h//2))
    return np.clip(warped, 0, 1), np.clip(target, 0, 1)


class FaceData(Dataset):
    def __init__(self, data_path):
        self.src = glob(f"{data_path}/src/aligned/*.jpg")
        self.dst = glob(f"{data_path}/dst/aligned/*.jpg")

    def __len__(self):
        return min(len(self.src), len(self.dst))

    def __getitem__(self, idx):
        s = np.asarray(Image.open(choice(self.src)).resize((192,192))) / 255.
        d = np.asarray(Image.open(choice(self.dst)).resize((192,192))) / 255.
        return s, d

    def collate_fn(self, batch):
        s, d = zip(*batch)
        ws, ts = get_training_data(s, len(s))
        wd, td = get_training_data(d, len(d))
        to_tensor = lambda x: torch.tensor(x, dtype=torch.float32).permute(0,3,1,2).to(device)
        return to_tensor(ws), to_tensor(ts), to_tensor(wd), to_tensor(td)


def pixel_norm(x, dim=-1):
    return x / torch.sqrt(torch.mean(x**2, dim=dim, keepdim=True) + 1e-6)

def depth_to_space(x, size=2):
    b, c, h, w = x.shape
    o_c = c // (size*size)
    x = x.reshape(b, size, size, o_c, h, w)
    x = x.permute(0,3,4,1,5,2).reshape(b, o_c, h*size, w*size)
    return x

class DepthToSpace(nn.Module):
    def forward(self, x, size=2): return depth_to_space(x, size)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,64,5,2,2), nn.LeakyReLU(0.1),
            nn.Conv2d(64,128,5,2,2), nn.LeakyReLU(0.1),
            nn.Conv2d(128,256,5,2,2), nn.LeakyReLU(0.1),
            nn.Conv2d(256,512,5,2,2), nn.LeakyReLU(0.1),
            nn.Flatten()
        )
    def forward(self,x): return pixel_norm(self.model(x))

class Upsample(nn.Module):
    def __init__(self, i, o): super().__init__()
    def forward(self,x): return DepthToSpace()(x)

class ResBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(c,c,3,1,1), nn.Conv2d(c,c,3,1,1)
    def forward(self,x):
        y = nn.functional.leaky_relu(self.c1(x),0.2)
        return nn.functional.leaky_relu(self.c2(y)+x,0.2)

class Inter(nn.Module):
    def __init__(self,dim=12800):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim,128), nn.Linear(128,1152)
        self.unflatten = nn.Unflatten(1,(128,3,3))
    def forward(self,x):
        if x.shape[1]!=self.fc1.in_features:
            self.fc1 = nn.Linear(x.shape[1],128).to(x.device)
        x=self.fc2(torch.relu(self.fc1(x)))
        return DepthToSpace()(self.unflatten(x))

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            ResBlock(128),
            nn.Conv2d(128,128,3,1,1),
            nn.LeakyReLU(0.1)
        )
        self.out = nn.Conv2d(128,3,3,1,1)
    def forward(self,x): return torch.sigmoid(self.out(self.seq(x)))


def create_window(size=11,sigma=1.5,channels=1):
    g=cv2.getGaussianKernel(size,sigma)
    gk=torch.tensor(g@ g.T,dtype=torch.float32)
    return gk.expand(channels,1,size,size).clone()

def dssim(i1,i2,window_size=11,eps=1e-6):
    pad=window_size//2
    w=create_window(window_size,channels=3).to(device)
    m1,m2=torch.conv2d(i1,w,padding=pad,groups=3),torch.conv2d(i2,w,padding=pad,groups=3)
    s1=torch.clamp(torch.conv2d(i1*i1,w,padding=pad,groups=3)-m1**2,min=0)
    s2=torch.clamp(torch.conv2d(i2*i2,w,padding=pad,groups=3)-m2**2,min=0)
    s12=torch.conv2d(i1*i2,w,padding=pad,groups=3)-m1*m2
    C1,C2=0.01**2,0.03**2
    ssim=((2*m1*m2+C1)*(2*s12+C2))/((m1**2+m2**2+C1)*(s1+s2+C2)+eps)
    return 1-torch.clamp(ssim.mean(),0,1)

#affichage 
def draw_results(src,t_src,dst,t_dst,fake,ls,ld):
    dpi=plt.rcParams['figure.dpi']
    fig,ax=plt.subplots(figsize=(660/dpi,370/dpi))
    ax.plot(ls,label='src'); ax.plot(ld,label='dst'); ax.legend()
    ax.set_title(f"Epoch {len(ls)}"); fig.canvas.draw()
    buf=np.frombuffer(fig.canvas.tostring_rgb(),dtype=np.uint8)
    img=buf.reshape(fig.canvas.get_width_height()[::-1]+(3,))/255.0
    plt.close(fig)
    grid=torchvision.utils.make_grid(
        [src[0],t_src[0],dst[0],t_dst[0],fake[0]],nrow=5,padding=10
    ).permute(1,2,0).cpu().numpy()
    grid=np.clip(grid,0,1)
    return np.vstack([img,grid])

#entrainement 
def train(data_path,model_name='Quick96',new_model=False,saved_models_dir='saved_model'):
    lr=5e-5; saved=Path(saved_models_dir)
    ds=FaceData(data_path)
    dl=DataLoader(ds,batch_size=16,shuffle=True,collate_fn=ds.collate_fn)
    enc,inter,dec_s,dec_d=Encoder().to(device),Inter().to(device),Decoder().to(device),Decoder().to(device)
    opt_e=torch.optim.Adam(list(enc.parameters())+list(inter.parameters()),lr=lr)
    opt_s=torch.optim.Adam(dec_s.parameters(),lr=lr)
    opt_d=torch.optim.Adam(dec_d.parameters(),lr=lr)
    mse=nn.MSELoss()

    epoch,ls,ld=0,[],[]
    model_file=saved/f"{model_name}.pth"
    if model_file.exists() and not new_model:
        ckpt=torch.load(model_file,map_location=device)
        enc.load_state_dict(ckpt['encoder']); inter.load_state_dict(ckpt['inter'])
        dec_s.load_state_dict(ckpt['decoder_src']); dec_d.load_state_dict(ckpt['decoder_dst'])
        opt_e.load_state_dict(ckpt['optimizer_encoder'])
        opt_s.load_state_dict(ckpt['optimizer_decoder_src'])
        opt_d.load_state_dict(ckpt['optimizer_decoder_dst'])
        ls,ld=ckpt['mean_loss_src'],ckpt['mean_loss_dst']
        epoch=ckpt['epoch']

    try:
        while True:
            epoch+=1
            e_ls,e_ld=[],[]
            for ws,ts,wd,td in tqdm(dl,desc=f"Epoch {epoch}"):
                # Source
                r_s=torch.clamp(dec_s(inter(enc(ws))),0,1)
                loss_s=0.8*mse(r_s,ts)+0.2*dssim(r_s,ts)
                opt_e.zero_grad(); opt_s.zero_grad()
                loss_s.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(),1.0)
                opt_e.step(); opt_s.step()

                # Dest
                r_d=torch.clamp(dec_d(inter(enc(wd))),0,1)
                loss_d=0.8*mse(r_d,td)+0.2*dssim(r_d,td)
                opt_e.zero_grad(); opt_d.zero_grad()
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(),1.0)
                opt_e.step(); opt_d.step()

                e_ls.append(loss_s.item()); e_ld.append(loss_d.item())

            ls.append(np.mean(e_ls)); ld.append(np.mean(e_ld))
            print(f"Epoch {epoch} | Lsrc={ls[-1]:.4f}, Ldst={ld[-1]:.4f}")

            # Affichage
            fake=dec_s(inter(enc(td)))
            res=draw_results(r_s,ts,r_d,td,fake,ls,ld)
            if HAS_DISPLAY:
                cv2.imshow('results',res); cv2.waitKey(1)
            else:
                Path("saved_results").mkdir(exist_ok=True)
                cv2.imwrite(f"saved_results/epoch_{epoch:04d}.jpg",(res*255).astype(np.uint8))

            # Sauvegarde toutes les 10 époques
            if epoch%10==0:
                ckpt={
                    'epoch':epoch,'encoder':enc.state_dict(),'inter':inter.state_dict(),
                    'decoder_src':dec_s.state_dict(),'decoder_dst':dec_d.state_dict(),
                    'optimizer_encoder':opt_e.state_dict(),'optimizer_decoder_src':opt_s.state_dict(),
                    'optimizer_decoder_dst':opt_d.state_dict(),'mean_loss_src':ls,'mean_loss_dst':ld
                }
                saved.mkdir(exist_ok=True,parents=True)
                torch.save(ckpt,model_file)
                print(f"[saved] epoch {epoch}")

    except KeyboardInterrupt:
        print("\n[stop] sauvegarde avant arrêt...")
        torch.save({
            'epoch':epoch,'encoder':enc.state_dict(),'inter':inter.state_dict(),
            'decoder_src':dec_s.state_dict(),'decoder_dst':dec_d.state_dict(),
            'optimizer_encoder':opt_e.state_dict(),'optimizer_decoder_src':opt_s.state_dict(),
            'optimizer_decoder_dst':opt_d.state_dict(),'mean_loss_src':ls,'mean_loss_dst':ld
        },model_file)
        print(f"Modèle sauvegardé : {model_file}")
        cv2.destroyAllWindows()
