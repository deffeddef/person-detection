"""
Video processing utilities.
"""
import cv2
from pathlib import Path
from .model import get_detector

def sample_frames(video_path: Path, every_n: int = 10):
    """
    A generator that yields frames from a video file.

    This function reads a video file and yields one frame every `every_n` frames.
    This is useful to avoid processing every single frame in a video, which can be
    time-consuming.

    Args:
        video_path: The path to the video file.
        every_n: The interval at which to sample frames. For example, if `every_n`
                 is 10, the function will yield the 0th, 10th, 20th, etc. frames.

    Yields:
        A video frame as a NumPy array.
    """
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
    """
    Predicts if a video contains multiple people.

    This function samples frames from a video and uses a face detector to count
    the number of faces in each frame. If any frame contains `min_faces` or more
    faces, the function immediately returns 1 (indicating multiple people). If the
    entire video is processed and no such frame is found, it returns 0.

    Args:
        video_path: The path to the video file.
        every_n: The interval at which to sample frames.
        conf: The confidence threshold for the face detector.
        min_faces: The minimum number of faces to detect in a frame to trigger a
                   positive prediction.

    Returns:
        1 if multiple people are detected, 0 otherwise.
    """
    detector = get_detector()
    # Iterate through sampled frames from the video
    for frame in sample_frames(video_path, every_n):
        # Detect faces in the current frame
        faces = detector.detect(frame, conf_threshold=conf)
        # If the number of detected faces meets the minimum requirement,
        # we can confidently say there are multiple people and stop early.
        if len(faces) >= min_faces:
            return 1
    # If we've gone through all the sampled frames and haven't found
    # enough faces in any of them, we conclude there's only one person (or none).
    return 0