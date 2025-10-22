import cv2
import numpy as np
import torch
from pathlib import Path
from face_extraction_tools import FaceExtractor
from face_masking import FaceMasking
from quick96 import Encoder, Inter, Decoder

# --- CONFIG ---
src_path = Path("data/data_src.jpg")   # visage B (celui qu'on veut coller)
dst_path = Path("data/data_dst.jpg")   # visage A (celui sur lequel on colle)
model_path = Path("saved_model/Quick96.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Chargement des images ---
src = cv2.imread(str(src_path))
dst = cv2.imread(str(dst_path))

h, w = dst.shape[:2]
face_extractor = FaceExtractor((w, h))
face_masker = FaceMasking()

# --- Détection des visages ---
ret_s, faces_s = face_extractor.detect(src)
ret_d, faces_d = face_extractor.detect(dst)

if faces_s is None or faces_d is None:
    raise RuntimeError("❌ Aucun visage détecté dans l'une des images.")

# --- Extraction alignée du visage source (B) ---
face_src, (x_s, y_s, w_s, h_s), M_s = face_extractor.extract(src, faces_s[0], desired_face_width=96)

# --- Extraction alignée du visage destination (A) ---
face_dst, (x_d, y_d, w_d, h_d), M_d = face_extractor.extract(dst, faces_d[0], desired_face_width=96)

# --- Chargement du modèle Quick96 ---
encoder, inter, decoder = Encoder().to(device), Inter().to(device), Decoder().to(device)
saved_model = torch.load(model_path, map_location=device)
encoder.load_state_dict(saved_model['encoder'])
inter.load_state_dict(saved_model['inter'])
decoder.load_state_dict(saved_model['decoder_src'])
model = torch.nn.Sequential(encoder, inter, decoder).eval()

# --- Inference ---
src_tensor = torch.tensor(face_src / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
with torch.no_grad():
    fake_torch = model(src_tensor)
fake_face = (fake_torch.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

# --- Masque ---
mask = face_masker.get_mask(fake_face)
mask = cv2.GaussianBlur(mask, (15, 15), 10)

# --- Transformation inverse pour replacer le visage fake dans A ---
M_inv = cv2.invertAffineTransform(M_d)
restored_face = cv2.warpAffine(fake_face, M_inv, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
restored_mask = cv2.warpAffine(mask, M_inv, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
restored_mask = (restored_mask > 1).astype(np.uint8) * 255

# --- Fusion avec seamlessClone ---
center = (x_d + w_d // 2, y_d + h_d // 2)
blended = cv2.seamlessClone(restored_face, dst, restored_mask, center, cv2.MIXED_CLONE)

# --- Sauvegarde et affichage ---
cv2.imwrite("data/face_swap_test.jpg", blended)
cv2.imshow("Résultat", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ Test terminé. Image enregistrée : data/face_swap_test.jpg")




import cv2
import numpy as np
from pathlib import Path
from face_masking import FaceMasking

# --- Images alignées ---
src_path = Path("data/src/aligned/00000.jpg")
dst_path = Path("data/dst/aligned/00000.jpg")
src = cv2.imread(str(src_path))
dst = cv2.imread(str(dst_path))

h, w = dst.shape[:2]
face_masker = FaceMasking()

# --- Test sans modèle (juste collage brut) ---
fake_face = src.copy()

# --- Création du masque ---
mask = face_masker.get_mask(fake_face)
mask = cv2.GaussianBlur(mask, (15, 15), 10)

# Debug : affichage du masque
cv2.imshow("mask", mask)
cv2.imwrite("data/mask_debug.jpg", mask)

# --- Centrage (au milieu de l'image) ---
center = (w // 2, h // 2)

# --- Fusion ---
try:
    blended = cv2.seamlessClone(fake_face, dst, mask, center, cv2.MIXED_CLONE)
except Exception as e:
    print(f"Erreur seamlessClone: {e}")
    blended = dst

cv2.imshow("source (src)", src)
cv2.imshow("destination (dst)", dst)
cv2.imshow("blended (swap)", blended)
cv2.imwrite("data/face_swap_test_debug.jpg", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2
import numpy as np
from pathlib import Path
from face_extraction_tools import FaceExtractor
from face_masking import FaceMasking

# -------------------------------------
# CONFIG
# -------------------------------------
src_path = Path("data/src/aligned/00000.jpg")  # visage source (B)
dst_path = Path("data/dst/aligned/00000.jpg")  # visage destination (A)
output_path = Path("data/face_swap_yunet_test.jpg")

# -------------------------------------
# LECTURE DES IMAGES
# -------------------------------------
src = cv2.imread(str(src_path))
dst = cv2.imread(str(dst_path))
if src is None or dst is None:
    raise FileNotFoundError("❌ Impossible de charger les images source ou destination.")

h, w = dst.shape[:2]
face_extractor = FaceExtractor((w, h))
face_masker = FaceMasking()

# -------------------------------------
# DÉTECTION DES VISAGES AVEC YUNET
# -------------------------------------
ret_s, faces_s = face_extractor.detect(src)
ret_d, faces_d = face_extractor.detect(dst)

if faces_s is None or len(faces_s) == 0:
    raise RuntimeError("❌ Aucun visage détecté dans la source.")
if faces_d is None or len(faces_d) == 0:
    raise RuntimeError("❌ Aucun visage détecté dans la destination.")

# -------------------------------------
# EXTRACTION / ALIGNEMENT
# -------------------------------------
face_src, (x_s, y_s, w_s, h_s), M_s = face_extractor.extract(src, faces_s[0], desired_face_width=128)
face_dst, (x_d, y_d, w_d, h_d), M_d = face_extractor.extract(dst, faces_d[0], desired_face_width=128)

# Debug visuel des visages alignés
cv2.imshow("face_src_aligned", face_src)
cv2.imshow("face_dst_aligned", face_dst)

# -------------------------------------
# COLLAGE DU VISAGE SOURCE SUR DESTINATION
# -------------------------------------
fake_face = face_src.copy()

# Créer le masque
mask = face_masker.get_mask(fake_face)
mask = cv2.GaussianBlur(mask, (15, 15), 10)

# Calcul de la transformation inverse du visage destination
M_inv = cv2.invertAffineTransform(M_d)

h_dst, w_dst = dst.shape[:2]
restored_face = cv2.warpAffine(fake_face, M_inv, (w_dst, h_dst), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
restored_mask = cv2.warpAffine(mask, M_inv, (w_dst, h_dst), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
restored_mask = (restored_mask > 1).astype(np.uint8) * 255

# -------------------------------------
# NORMALISATION DES TAILLES (sécurité)
# -------------------------------------
# Si les tailles diffèrent -> on recentre plutôt que d'étirer
if restored_face.shape[:2] != dst.shape[:2]:
    canvas = np.zeros_like(dst)
    mask_canvas = np.zeros((h_dst, w_dst), dtype=np.uint8)

    y0 = (h_dst - restored_face.shape[0]) // 2
    x0 = (w_dst - restored_face.shape[1]) // 2
    canvas[y0:y0+restored_face.shape[0], x0:x0+restored_face.shape[1]] = restored_face
    mask_canvas[y0:y0+restored_mask.shape[0], x0:x0+restored_mask.shape[1]] = restored_mask

    restored_face = canvas
    restored_mask = mask_canvas

# S'assurer que le masque est 1 canal et uint8
if restored_mask.ndim == 3:
    restored_mask = cv2.cvtColor(restored_mask, cv2.COLOR_BGR2GRAY)
restored_mask = np.clip(restored_mask, 0, 255).astype(np.uint8)

# Centre du collage (milieu de la zone du visage destination)
center = (x_d + w_d // 2, y_d + h_d // 2)

# -------------------------------------
# FUSION (cv2.seamlessClone)
# -------------------------------------
try:
    blended = cv2.seamlessClone(restored_face, dst, restored_mask, center, cv2.NORMAL_CLONE)
except Exception as e:
    print(f"⚠️ Erreur lors du seamlessClone : {e}")
    blended = dst

# -------------------------------------
# DEBUG VISUEL ET SAUVEGARDE
# -------------------------------------
concat = np.hstack([
    cv2.resize(src, (w_dst, h_dst)),
    cv2.resize(dst, (w_dst, h_dst)),
    blended
])

cv2.imshow("YuNet Face Swap Debug (SRC | DST | RESULT)", concat)
cv2.imwrite(str(output_path), blended)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"✅ Test réussi ! Image sauvegardée dans : {output_path}")



