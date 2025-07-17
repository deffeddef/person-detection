
import cv2
from pathlib import Path
from .model import get_detector

def sample_frames(video_path: Path, every_n: int = 10):
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            yield frame
        idx += 1
    cap.release()

def predict_video(video_path: Path, every_n: int = 10, conf: float = 0.5, min_faces: int = 2):
    detector = get_detector()
    for frame in sample_frames(video_path, every_n):
        faces = detector.detect(frame, conf_threshold=conf)
        if len(faces) >= min_faces:
            return 1
    return 0
