import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from quick96 import Encoder, Inter, Decoder  # ton modèle Quick96

device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'


def merge_frames_to_fake_video(dst_frames_path, model_name='Quick96', saved_models_dir='saved_model'):
    dst_frames_path = Path(dst_frames_path)
    model_path = Path(saved_models_dir) / f"{model_name}.pth"

    # --- Charger le modèle ---
    encoder = Encoder().to(device)
    inter = Inter().to(device)
    decoder = Decoder().to(device)
    state_dict = torch.load(model_path, map_location=device)
    encoder.load_state_dict(state_dict['encoder'])
    inter.load_state_dict(state_dict['inter'])
    decoder.load_state_dict(state_dict['decoder'])
    encoder.eval(); inter.eval(); decoder.eval()

    # --- Obtenir la taille de la première frame ---
    first_frame_path = next(dst_frames_path.glob('*.jpg'))
    frame0 = cv2.imread(str(first_frame_path))
    height, width = frame0.shape[:2]

    result_path = dst_frames_path.parent / "deepfake_result.mp4"
    video_writer = cv2.VideoWriter(
        str(result_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        25.0,
        (width, height)
    )

    # --- Parcours des frames ---
    face_files = sorted(dst_frames_path.glob('*.jpg'))
    for face_path in tqdm(face_files, desc="Merging faces"):
        frame = cv2.imread(str(face_path))
        if frame is None:
            continue

        # nom du visage et matrice associée
        face_name = face_path.stem
        M_path = face_path.with_suffix('.npy')
        if not M_path.exists():
            continue

        M = np.load(M_path)
        M_inv = cv2.invertAffineTransform(M)

        # Charger visage aligné 256x256 (ici, on suppose que tu as la reconstruction Quick96)
        aligned_face = cv2.imread(str(face_path))
        aligned_face = cv2.resize(aligned_face, (256, 256))
        aligned_face = aligned_face.astype(np.float32) / 255.0
        face_tensor = torch.from_numpy(aligned_face.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            latent = encoder(face_tensor)
            latent = inter(latent)
            pred_face = decoder(latent).cpu().numpy()[0].transpose(1, 2, 0)

        pred_face = np.clip(pred_face * 255, 0, 255).astype(np.uint8)

        # --- Reprojection du visage sur la frame originale ---
        reprojected_face = cv2.warpAffine(pred_face, M_inv, (width, height), borderMode=cv2.BORDER_REFLECT)

        # --- Masque pour blending doux ---
        mask = np.ones((256, 256), dtype=np.float32)
        mask = cv2.GaussianBlur(mask, (61, 61), 30)
        mask = cv2.warpAffine(mask, M_inv, (width, height))
        mask = np.expand_dims(mask, axis=2)

        blended = frame * (1 - mask) + reprojected_face * mask
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        video_writer.write(blended)

    video_writer.release()
    print(f"[OK] Deepfake vidéo sauvegardée : {result_path}")

