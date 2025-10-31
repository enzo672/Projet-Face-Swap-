import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
from face_extraction_tools import FaceExtractor
from face_masking import FaceMasking
from quick96 import Encoder, Inter, Decoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def merge_frames_to_fake_video(src_face_path, dst_frames_path, model_name='Quick96', saved_models_dir='saved_model'):
    model_path = Path(saved_models_dir) / f'{model_name}.pth'
    dst_frames_path = Path(dst_frames_path)
    image = Image.open(next(dst_frames_path.glob('*.jpg')))
    frame_w, frame_h = image.size
    result_video = cv2.VideoWriter(
        str(dst_frames_path.parent / 'fake.mp4'),
        cv2.VideoWriter_fourcc(*'MJPG'),
        30,
        (frame_w, frame_h)
    )

    face_extractor = FaceExtractor((frame_w, frame_h))
    face_masker = FaceMasking()

    # Chargement du modèle
    encoder = Encoder().to(device)
    inter = Inter().to(device)
    decoder_src = Decoder().to(device)
    decoder_dst = Decoder().to(device)

    saved_model = torch.load(model_path, map_location=device)
    encoder.load_state_dict(saved_model['encoder'])
    inter.load_state_dict(saved_model['inter'])
    decoder_src.load_state_dict(saved_model['decoder_src'])
    decoder_dst.load_state_dict(saved_model['decoder_dst'])

    encoder.eval()
    inter.eval()
    decoder_src.eval()
    decoder_dst.eval()

    # Préparation du visage source
    src_face = cv2.imread(str(src_face_path))
    _, src_bbox = face_extractor.detect(src_face)
    if src_bbox is None:
        raise ValueError("Aucun visage détecté dans l'image source.")
    src_crop, _ = face_extractor.extract(src_face, src_bbox[0])
    src_crop = cv2.resize(src_crop, (96, 96))[..., ::-1]
    src_tensor = torch.tensor(src_crop / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        latent_src = inter(encoder(src_tensor))

    # Application frame par frame
    frames_list = sorted(dst_frames_path.glob('*.jpg'))
    for i, frame_path in enumerate(frames_list, 1):
        print(f'Frame {i}/{len(frames_list)}')
        frame = cv2.imread(str(frame_path))
        retval, dst_faces = face_extractor.detect(frame)
        if not dst_faces:
            result_video.write(frame)
            continue

        dst_face_crop, dst_bbox = face_extractor.extract(frame, dst_faces[0])
        dst_face_crop_rgb = dst_face_crop[..., ::-1]
        dst_face_crop_rgb = cv2.resize(dst_face_crop_rgb, (96, 96))

        with torch.no_grad():
            fake_torch = decoder_dst(latent_src)
            fake_face = (fake_torch.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        mask_fake = face_masker.get_mask(fake_face)
        mask_fake = cv2.resize(mask_fake, (dst_face_crop.shape[1], dst_face_crop.shape[0]))

        M = cv2.moments(mask_fake)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else dst_bbox[0] + dst_bbox[2] // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else dst_bbox[1] + dst_bbox[3] // 2

        try:
            blended_face = cv2.seamlessClone(
                fake_face[..., ::-1],
                dst_face_crop,
                mask_fake,
                (cx, cy),
                cv2.NORMAL_CLONE
            )
        except cv2.error:
            result_video.write(frame)
            continue

        x, y, w, h = dst_bbox
        blended_resized = cv2.resize(blended_face, (w, h))
        frame[y:y+h, x:x+w] = blended_resized
        result_video.write(frame)

    result_video.release()
    print("Fake video saved:", dst_frames_path.parent / 'fake.mp4')
