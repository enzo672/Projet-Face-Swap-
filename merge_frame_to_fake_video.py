from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from face_extraction_tools import FaceExtractor
from face_masking import FaceMasking
from quick96 import Encoder, Inter, Decoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def merge_frames_to_fake_video(dst_frames_path, model_name='Quick96', saved_models_dir='saved_model'):
    dst_frames_path = Path(dst_frames_path)
    model_path = Path(saved_models_dir) / f'{model_name}.pth'

    # Taille vidéo à partir d’une image
    first_image_path = next(dst_frames_path.glob('*.jpg'))
    first_frame = cv2.imread(str(first_image_path))
    h, w = first_frame.shape[:2]

    result_video = cv2.VideoWriter(
        str(dst_frames_path.parent / 'fake.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (w, h)
    )

    # Initialisation
    face_extractor = FaceExtractor((w, h))
    face_masker = FaceMasking()

    # Chargement du modèle Quick96
    encoder, inter, decoder = Encoder().to(device), Inter().to(device), Decoder().to(device)
    saved_model = torch.load(model_path, map_location=device)
    encoder.load_state_dict(saved_model['encoder'])
    inter.load_state_dict(saved_model['inter'])
    decoder.load_state_dict(saved_model['decoder_src'])
    model = torch.nn.Sequential(encoder, inter, decoder).eval()

    frames_list = sorted(dst_frames_path.glob('*.jpg'))

    for ii, frame_path in enumerate(tqdm(frames_list, desc="Merging frames")):
        frame = cv2.imread(str(frame_path))
        ret, faces = face_extractor.detect(frame)
        if faces is None or len(faces) == 0:
            result_video.write(frame)
            continue

        # On prend le premier visage
        face = faces[0]
        aligned_face, (x, y, w_face, h_face), M = face_extractor.extract(frame, face, desired_face_width=96)
        if aligned_face is None:
            result_video.write(frame)
            continue

        # Inférence
        face_tensor = torch.tensor(aligned_face / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            generated_torch = model(face_tensor)
        generated_face = (generated_torch.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Masque et flou
        mask = face_masker.get_mask(generated_face)
        mask = cv2.GaussianBlur(mask, (15, 15), 10)

        # --- Appliquer la transformation inverse pour replacer ---
        M_inv = cv2.invertAffineTransform(M)
        restored_face = cv2.warpAffine(generated_face, M_inv, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        restored_mask = cv2.warpAffine(mask, M_inv, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)

        # Normalisation du masque
        restored_mask = (restored_mask > 1).astype(np.uint8) * 255

        # Fusion douce dans l’image originale
        center = (w // 2, h // 2)
        try:
            blended = cv2.seamlessClone(restored_face, frame, restored_mask, center, cv2.MIXED_CLONE)
        except Exception as e:
            print(f"⚠️ Seamless clone failed on frame {ii}: {e}")
            blended = frame

        result_video.write(blended)

    result_video.release()
    print("✅ Fusion terminée : fake.mp4 généré avec succès")
