from pathlib import Path
import cv2

import face_extraction_tools as fet
import quick96 as q96
from merge_frame_to_fake_video import merge_frames_to_fake_video

extract_and_align_src = True
extract_and_align_dst = True
train = False
eval = True

model_name = 'Quick96'
new_model = False

def sanity_check_opencv():
    print("[setup] OpenCV version:", cv2.__version__)
    if not hasattr(cv2, "FaceDetectorYN"):
        print("[warning] cv2.FaceDetectorYN introuvable. "
              "Installe opencv-contrib-python (ex: pip install 'opencv-contrib-python==4.9.0.80').")


project_root = Path(__file__).parent
data_root = project_root / 'data'
src_video_path = data_root / 'data_src.mp4'
dst_video_path = data_root / 'data_dst.mp4'

src_processing_folder = data_root / 'src'
dst_processing_folder = data_root / 'dst'

models_dir = project_root / "models"         
saved_models_dir = project_root / "saved_model"


sanity_check_opencv()

# étape 1: extraire les frames 
if extract_and_align_src:
    fet.extract_frames_from_video(
        video_path=src_video_path,
        output_folder=src_processing_folder,
        frames_to_skip=0
    )
if extract_and_align_dst:
    fet.extract_frames_from_video(
        video_path=dst_video_path,
        output_folder=dst_processing_folder,
        frames_to_skip=0
    )

# étape 2: extraction et alignement des visages 
if extract_and_align_src:
    fet.extract_and_align_face_from_image(
        input_dir=src_processing_folder,
        desired_face_width=256
    )
if extract_and_align_dst:
    fet.extract_and_align_face_from_image(
        input_dir=dst_processing_folder,
        desired_face_width=256
    )

# étape 3: model training 
if train:
    q96.train(str(data_root), model_name, new_model, saved_models_dir=str(saved_models_dir))

# étape 4: fake video 
if eval:
    merge_frames_to_fake_video(dst_processing_folder, model_name, saved_models_dir=str(saved_models_dir))
