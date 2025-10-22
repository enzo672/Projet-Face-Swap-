import cv2
import numpy as np
import torch
from pathlib import Path
from face_extraction_tools import FaceExtractor
from face_masking import FaceMasking
from quick96 import Encoder, Inter, Decoder

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
src_path = Path("data/src/aligned/00000.jpg")  # visage à transférer (B)
dst_path = Path("data/dst/aligned/00000.jpg")  # visage cible (A)
model_path = Path("saved_model/Quick96.pth")
output_path = Path("data/face_swap_final_test.jpg")

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# LECTURE DES IMAGES
# -------------------------------------------------
src = cv2.imread(str(src_path))
dst = cv2.imread(str(dst_path))
if src is None or dst is None:
    raise FileNotFoundError("❌ Impossible de charger les images source ou destination.")

h, w = dst.shape[:2]
face_extractor = FaceExtractor((w, h))
face_masker = FaceMasking()

# -------------------------------------------------
# DÉTECTION DES VISAGES AVEC YUNET
# -------------------------------------------------
ret_s, faces_s = face_extractor.detect(src)
ret_d, faces_d = face_extractor.detect(dst)

if faces_s is None or len(faces_s) == 0:
    raise RuntimeError("❌ Aucun visage détecté dans la source.")
if faces_d is None or len(faces_d) == 0:
    raise RuntimeError("❌ Aucun visage détecté dans la destination.")

# -------------------------------------------------
# EXTRACTION / ALIGNEMENT
# -------------------------------------------------
face_src, (x_s, y_s, w_s, h_s), M_s = face_extractor.extract(src, faces_s[0], desired_face_width=128)
face_dst, (x_d, y_d, w_d, h_d), M_d = face_extractor.extract(dst, faces_d[0], desired_face_width=128)

# -------------------------------------------------
# CHARGEMENT DU MODÈLE QUICK96
# -------------------------------------------------
encoder, inter, decoder = Encoder().to(device), Inter().to(device), Decoder().to(device)
saved_model = torch.load(model_path, map_location=device)
encoder.load_state_dict(saved_model["encoder"])
inter.load_state_dict(saved_model["inter"])
decoder.load_state_dict(saved_model["decoder_src"])
model = torch.nn.Sequential(encoder, inter, decoder).eval()

# -------------------------------------------------
# GÉNÉRATION DU VISAGE FAKE
# -------------------------------------------------
src_tensor = torch.tensor(face_src / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
with torch.no_grad():
    fake_torch = model(src_tensor)
fake_face = (fake_torch.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

# -------------------------------------------------
# MASQUE
# -------------------------------------------------
mask = face_masker.get_mask(fake_face)
mask = cv2.GaussianBlur(mask, (15, 15), 10)

# -------------------------------------------------
# TRANSFORMATION INVERSE POUR REPLACER LE VISAGE
# -------------------------------------------------
M_inv = cv2.invertAffineTransform(M_d)

h_dst, w_dst = dst.shape[:2]
restored_face = cv2.warpAffine(fake_face, M_inv, (w_dst, h_dst), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
restored_mask = cv2.warpAffine(mask, M_inv, (w_dst, h_dst), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
restored_mask = (restored_mask > 1).astype(np.uint8) * 255

# -------------------------------------------------
# NORMALISATION DES FORMATS / TAILLES
# -------------------------------------------------
# Étend à la même taille que dst si besoin
if restored_face.shape[:2] != dst.shape[:2]:
    restored_face = cv2.resize(restored_face, (dst.shape[1], dst.shape[0]), interpolation=cv2.INTER_CUBIC)
if restored_mask.shape[:2] != dst.shape[:2]:
    restored_mask = cv2.resize(restored_mask, (dst.shape[1], dst.shape[0]), interpolation=cv2.INTER_NEAREST)

# Conversion en format correct
if restored_mask.ndim == 3:
    restored_mask = cv2.cvtColor(restored_mask, cv2.COLOR_BGR2GRAY)
restored_mask = np.clip(restored_mask, 0, 255).astype(np.uint8)
restored_face = restored_face.astype(np.uint8)
dst = dst.astype(np.uint8)

center = (x_d + w_d // 2, y_d + h_d // 2)

# -------------------------------------------------
# FUSION FINALE
# -------------------------------------------------
try:
    blended = cv2.seamlessClone(restored_face, dst, restored_mask, center, cv2.NORMAL_CLONE)
except Exception as e:
    print(f"⚠️ Erreur seamlessClone : {e}")
    blended = dst

# -------------------------------------------------
# SAUVEGARDE ET VISUALISATION
# -------------------------------------------------
cv2.imwrite(str(output_path), blended)

concat = np.hstack([
    cv2.resize(src, (w_dst, h_dst)),
    cv2.resize(dst, (w_dst, h_dst)),
    blended
])

cv2.imshow("Quick96 Face Swap (SRC | DST | RESULT)", concat)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"✅ Swap réussi ! Image sauvegardée : {output_path}")
