from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
from face_extraction_tools import FaceExtractor
from face_masking import FaceMasking
from quick96 import Encoder, Inter, Decoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def merge_frames_to_fake_video(dst_frames_path, model_name='Quick96', saved_models_dir='saved_model', output_name='fake.mp4', fps=30):
    """
    Merge destination frames into a swapped-face video using the trained Quick96 model.
    
    Args:
        dst_frames_path (str|Path): folder containing destination frames (.jpg)
        model_name (str): model filename (without extension)
        saved_models_dir (str|Path): directory containing saved .pth model
        output_name (str): name of output video
        fps (int): frames per second for the result video
    """
    dst_frames_path = Path(dst_frames_path)
    model_path = Path(saved_models_dir) / f'{model_name}.pth'
    assert dst_frames_path.exists(), f"[error] Destination frames path not found: {dst_frames_path}"
    assert model_path.exists(), f"[error] Model not found: {model_path}"

    # Load first frame to get video size
    first_image_path = next(dst_frames_path.glob('*.jpg'), None)
    if first_image_path is None:
        raise ValueError(f"No .jpg frames found in {dst_frames_path}")
    first_image = Image.open(first_image_path)
    image_size = first_image.size  # (W, H)

    # Prepare output video writer
    result_video_path = dst_frames_path.parent / output_name
    result_video = cv2.VideoWriter(str(result_video_path), cv2.VideoWriter_fourcc(*'MJPG'), fps, image_size)

    print(f"[info] Loading model from {model_path}")
    saved_model = torch.load(model_path, map_location=device)

    # Load trained networks
    encoder = Encoder().to(device)
    inter = Inter().to(device)
    decoder_src = Decoder().to(device)
    decoder_dst = Decoder().to(device)

    encoder.load_state_dict(saved_model['encoder'])
    inter.load_state_dict(saved_model['inter'])
    decoder_src.load_state_dict(saved_model['decoder_src'])
    decoder_dst.load_state_dict(saved_model['decoder_dst'])

    encoder.eval()
    inter.eval()
    decoder_src.eval()
    decoder_dst.eval()

    print(f"[info] Model loaded. Starting face swap on frames from: {dst_frames_path}")
    face_extractor = FaceExtractor(image_size)
    face_masker = FaceMasking()

    frames_list = sorted(dst_frames_path.glob('*.jpg'))
    total = len(frames_list)
    for ii, frame_path in enumerate(frames_list, 1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"[warn] Cannot read {frame_path}, skipping.")
            continue

        retval, faces = face_extractor.detect(frame)
        if not retval or len(faces) == 0:
            result_video.write(frame)
            continue

        # take the largest detected face
        face = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
        face_image, (x, y, w, h) = face_extractor.extract(frame, face)
        face_image = face_image[..., ::-1].astype(np.float32) / 255.0  # BGR->RGB, normalize

        # Resize to model input size (96x96)
        face_cropped = cv2.resize(face_image, (96, 96))
        face_t = torch.tensor(face_cropped).permute(2, 0, 1).unsqueeze(0).to(device)

        # ----- SWAP -----
        latent = inter(encoder(face_t))
        generated_face_torch = decoder_src(latent)
        generated_face = (generated_face_torch.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Build blending masks
        mask_origin = face_masker.get_mask((face_cropped * 255).astype(np.uint8))
        mask_fake = face_masker.get_mask(generated_face)

        # Find clone center for seamless blending
        m = cv2.moments(mask_origin)
        if m['m00'] == 0:
            result_video.write(frame)
            continue
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])

        try:
            blended_face = cv2.seamlessClone(generated_face, (face_cropped * 255).astype(np.uint8),
                                             mask_fake, (cx, cy), cv2.NORMAL_CLONE)
        except Exception as e:
            print(f"[warn] Skip frame {ii}/{total}: blending failed ({e})")
            result_video.write(frame)
            continue

        # Resize to original face ROI
        blended_face_bgr = cv2.resize(blended_face[..., ::-1], (w, h))  # back to BGR
        frame[y:y + h, x:x + w] = blended_face_bgr

        result_video.write(frame)
        if ii % 20 == 0:
            print(f"[progress] Processed {ii}/{total} frames")

    result_video.release()
    print(f"[done] Swapped video saved at: {result_video_path}")
    return str(result_video_path)


if __name__ == "__main__":
    # Example usage
    output = merge_frames_to_fake_video(
        dst_frames_path="data/dst/aligned",
        model_name="Quick96",
        saved_models_dir="saved_model",
        output_name="fake.mp4",
        fps=30
    )
    print("Output video:", output)
