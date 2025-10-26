import torch
from pathlib import Path
import cv2
import numpy as np
from quick96 import Encoder, Inter, Decoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = 'Quick96'
saved_models_dir = Path('saved_model')
src_face_path = Path('data/src/aligned/00000.jpg')   # visage à copier (brune)
dst_face_path = Path('data/dst/aligned/00000.jpg')   # visage cible (blonde)
output_path = Path('data/output_swap_inverted.jpg')

# Chargement du modèle 
encoder = Encoder().to(device)
inter = Inter().to(device)
decoder_A = Decoder().to(device)  # pour SRC
decoder_B = Decoder().to(device)  # pour DST

model_path = saved_models_dir / f'{model_name}.pth'
if not model_path.exists():
    raise FileNotFoundError(f" Modèle non trouvé : {model_path}")

checkpoint = torch.load(model_path, map_location=device)

# Dans l'entrainement les decodeur ont été inversés 
encoder.load_state_dict(checkpoint['encoder'])
inter.load_state_dict(checkpoint['inter'])
decoder_A.load_state_dict(checkpoint['decoder_dst'])  # A = dst
decoder_B.load_state_dict(checkpoint['decoder_src'])  # B = src

encoder.eval(); inter.eval(); decoder_A.eval(); decoder_B.eval()
print(" Modèle Quick96 chargé")

# Chargement visages alignés 96x96 
def load_face(p):
    im_bgr = cv2.imread(str(p))
    if im_bgr is None:
        raise FileNotFoundError(f" Image non trouvée : {p}")
    im_bgr = cv2.resize(im_bgr, (96, 96))
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    t = torch.tensor(im_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return im_bgr, t.to(device)

src_bgr, src_t = load_face(src_face_path)
dst_bgr, dst_t = load_face(dst_face_path)

#Swap 
with torch.no_grad():
    latent_src = encoder(src_t)
    inter_latent = inter(latent_src)
    swapped_face = decoder_B(inter_latent)  

# Convertir en image
swapped_face = swapped_face.squeeze(0).permute(1, 2, 0).cpu().numpy()
swapped_face = np.clip(swapped_face * 255, 0, 255).astype(np.uint8)
swapped_bgr = cv2.cvtColor(swapped_face, cv2.COLOR_RGB2BGR)

# Etape de fusion 

# Réduction légère
scale_factor = 0.9
center = (48, 48)
M = cv2.getRotationMatrix2D(center, 0, scale_factor)
swapped_resized = cv2.warpAffine(swapped_bgr, M, (96, 96), flags=cv2.INTER_CUBIC)

# Masque plus petit
mask = np.zeros((96, 96), dtype=np.uint8)
cv2.circle(mask, (48, 48), 30, 255, -1)
mask = cv2.GaussianBlur(mask, (7, 7), 3)

# Fusion Poisson
blended_poisson = cv2.seamlessClone(
    swapped_resized,
    dst_bgr,
    mask,
    (48, 48),
    cv2.NORMAL_CLONE
)

alpha = (mask.astype(np.float32) / 255.0)[..., None]
blended = (alpha * blended_poisson + (1 - alpha) * dst_bgr).astype(np.uint8)

#Sauvegarde/affichage 
comparison = np.hstack([
    cv2.resize(src_bgr, (96, 96)),
    cv2.resize(dst_bgr, (96, 96)),
    blended
])

cv2.imwrite(str(output_path), blended)
cv2.imwrite("data/comparison_swap_inverted.jpg", comparison)

print(f" Swap A vers B  terminé → {output_path}")

cv2.imshow("Source (A) | Cible (B) | Swap A→B", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()
