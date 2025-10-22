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
    model_path = Path(saved_models_dir) / f'{model_name}.pth'
    dst_frames_path = Path(dst_frames_path)

    # Récupération d'une image pour définir la taille de la vidéo
    first_image = cv2.imread(str(next(dst_frames_path.glob('*.jpg'))))
    h, w = first_image.shape[:2]
    result_video = cv2.VideoWriter(
        str(dst_frames_path.parent / 'fake.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (w, h)
    )

    # Initialisation des outils
    face_extractor = FaceExtractor()
    face_masker = FaceMasking()

    # Chargement du modèle
    encoder, inter, decoder = Encoder().to(device), Inter().to(device), Decoder().to(device)
    saved_model = torch.load(model_path)
    encoder.load_state_dict(saved_model['encoder'])
    inter.load_state_dict(saved_model['inter'])
    decoder.load_state_dict(saved_model['decoder_src'])
    model = torch.nn.Sequential(encoder, inter, decoder)

    frames_list = sorted(dst_frames_path.glob('*.jpg'))

    for ii, frame_path in enumerate(tqdm(frames_list, desc="Merging frames")):
        frame = cv2.imread(str(frame_path))
        retval, faces = face_extractor.detect(frame)
        if faces is None or len(faces) == 0:
            result_video.write(frame)
            continue

        # On prend le premier visage détecté
        x, y, w_face, h_face = faces[0]
        face_crop = frame[y:y+h_face, x:x+w_face]

        # --- Prétraitement pour le modèle ---
        resized_face = cv2.resize(face_crop, (96, 96))
        input_tensor = torch.tensor(resized_face / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        # --- Génération ---
        with torch.no_grad():
            generated = model(input_tensor)
        generated_face = (generated.squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

        # --- Masques pour fusion ---
        mask = face_masker.get_mask(generated_face)
        mask = cv2.GaussianBlur(mask, (15,15), 10)

        # --- Remise à l’échelle et repositionnement ---
        generated_resized = cv2.resize(generated_face, (w_face, h_face))
        mask_resized = cv2.resize(mask, (w_face, h_face))

        center = (x + w_face // 2, y + h_face // 2)

        # --- Clone harmonieux ---
        try:
            blended = cv2.seamlessClone(generated_resized, frame, mask_resized, center, cv2.NORMAL_CLONE)
        except:
            blended = frame

        result_video.write(blended)

    result_video.release()
    print("✅ Video fusionnée : fake.mp4")
