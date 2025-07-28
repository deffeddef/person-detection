import pytest
from pathlib import Path
import cv2
import numpy as np
from src.utils import predict_video, sample_frames

@pytest.fixture
def dummy_video():
    """Creates a dummy video file with 30 blank frames and yields its path."""
    tmp_path = Path('tests/_blank.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(tmp_path), fourcc, 10, (640, 480))
    for _ in range(30):
        out.write(np.zeros((480, 640, 3), dtype=np.uint8))
    out.release()
    yield tmp_path
    tmp_path.unlink()

def test_sample_frames(dummy_video):
    """Tests the frame sampling logic of the sample_frames generator."""
    # Test with every_n = 10, expecting 3 frames (0, 10, 20)
    frames_n_10 = list(sample_frames(dummy_video, every_n=10))
    assert len(frames_n_10) == 3

    # Test with every_n = 5, expecting 6 frames (0, 5, 10, 15, 20, 25)
    frames_n_5 = list(sample_frames(dummy_video, every_n=5))
    assert len(frames_n_5) == 6

    # Test with every_n = 1, expecting all 30 frames
    frames_n_1 = list(sample_frames(dummy_video, every_n=1))
    assert len(frames_n_1) == 30

def test_predict_video_dummy(dummy_video):
    """Tests predict_video on a blank video, expecting no faces found."""
    # Since the video is blank, the prediction should be 0 (no multiple faces)
    label = predict_video(dummy_video, every_n=1)
    assert label == 0

def test_predict_video_positive():
    """
    Tests predict_video on a video known to have multiple faces.
    Note: This is an integration test and requires 'tests/test_video_positive.mp4'.
    """
    video_path = Path('tests/test_video_positive.mp4')
    if not video_path.exists():
        pytest.skip("Test video not found: test_video_positive.mp4")
    label = predict_video(video_path, every_n=1)
    assert label == 1

def test_predict_video_negative():
    """
    Tests predict_video on a video known to have no multiple faces.
    Note: This is an integration test and requires 'tests/test_video_negative.mp4'.
    """
    video_path = Path('tests/test_video_negative.mp4')
    if not video_path.exists():
        pytest.skip("Test video not found: test_video_negative.mp4")
    label = predict_video(video_path, every_n=1)
    assert label == 0