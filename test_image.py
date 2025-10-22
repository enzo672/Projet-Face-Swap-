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
