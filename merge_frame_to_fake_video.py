from pathlib import Path
import torch
import cv2
import numpy as np
from face_extraction_tools import FaceExtractor
from face_masking import FaceMasking
from quick96 import Encoder, Inter, Decoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def merge_frames_to_fake_video(src_image_path, dst_frames_path, model_name='Quick96', saved_models_dir='saved_model'):
    """
    Génère une vidéo deepfake dynamique :
    identité de la source (A) sur les mouvements/expressions de la destination (B).
    """

    # Préparation des chemins
    model_path = Path(saved_models_dir) / f'{model_name}.pth'
    dst_frames_path = Path(dst_frames_path)
    src_path = Path(src_image_path)

    print(f"[INFO] Chargement du modèle : {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    # Chargement du modèle Quick96
    encoder = Encoder().to(device)
    inter = Inter().to(device)
    decoder_src = Decoder().to(device)
    decoder_dst = Decoder().to(device)

    saved_model = torch.load(model_path, map_location=device)
    encoder.load_state_dict(saved_model['encoder'])
    inter.load_state_dict(saved_model['inter'])
    decoder_src.load_state_dict(saved_model['decoder_src'])
    decoder_dst.load_state_dict(saved_model['decoder_dst'])
    encoder.eval(); inter.eval(); decoder_src.eval(); decoder_dst.eval()

    # Chargement visage source (identité à transférer)
    if src_path.is_dir():
        src_images = sorted(src_path.glob("*.jpg")) + sorted(src_path.glob("*.png"))
        if not src_images:
            raise FileNotFoundError(f"Aucune image trouvée dans {src_path}")
        src_face_path = src_images[0]
    else:
        src_face_path = src_path

    print(f"[INFO] Image source utilisée : {src_face_path}")
    src_face = cv2.imread(str(src_face_path))
    if src_face is None:
        raise FileNotFoundError(f"Impossible de lire l'image source : {src_face_path}")

    src_face = cv2.resize(src_face, (96, 96))
    src_tensor = torch.tensor(src_face[..., ::-1]/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # Préparation du writer vidéo
    dst_images = sorted(dst_frames_path.glob("*.jpg"))
    if not dst_images:
        raise FileNotFoundError(f"Aucune image .jpg trouvée dans {dst_frames_path}")

    first_frame = cv2.imread(str(dst_images[0]))
    if first_frame is None:
        raise FileNotFoundError(f"Impossible de lire la première frame dans {dst_frames_path}")

    h, w = first_frame.shape[:2]
    output_path = dst_frames_path.parent / 'fake_swap_dynamic.mp4'
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    print(f"[INFO] Vidéo de sortie : {output_path}")

    face_extractor = FaceExtractor((w, h))
    face_masker = FaceMasking()

    # Boucle principale sur les frames destination
    total_frames = len(dst_images)
    for i, frame_path in enumerate(dst_images, 1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"[WARN] Impossible de lire {frame_path}, ignorée.")
            continue

        retval, face = face_extractor.detect(frame)
        if face is None:
            out.write(frame)
            continue

        face_image, box = face_extractor.extract(frame, face[0])
        if face_image is None or face_image.size == 0:
            print(f"[WARN] Aucun visage extrait sur {frame_path}")
            out.write(frame)
            continue

        face_image = cv2.resize(face_image, (96, 96))
        face_tensor = torch.tensor(face_image[..., ::-1]/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        # encode chaque frame destination (pose/expression) → decode avec le décodeur source (identité)
        with torch.no_grad():
            latent_dst = encoder(face_tensor)
            latent_dst = inter(latent_dst)
            fake_face = decoder_src(latent_dst)  # décode avec l'identité source

        fake_face_np = (fake_face.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Application du masque et recollement
        mask = face_masker.get_mask(fake_face_np)
        try:
            output_face = cv2.seamlessClone(fake_face_np, face_image, mask, (48, 48), cv2.NORMAL_CLONE)
        except Exception as e:
            print(f"[WARN] Clone échoué sur {frame_path}: {e}")
            out.write(frame)
            continue

        frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = cv2.resize(output_face, (box[2], box[3]))
        out.write(frame)

        if i % 10 == 0:
            print(f"[INFO] Frame {i}/{total_frames} traitée")

    out.release()
    print(f"[SUCCESS] Vidéo dynamique générée : {output_path}")
