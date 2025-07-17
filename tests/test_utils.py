
from pathlib import Path
from veriff_multiperson_detector.src.utils import predict_video

def test_dummy():
    import cv2
    import numpy as np
    tmp = Path('tests/_blank.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(tmp), fourcc, 10, (640, 480))
    for _ in range(10):
        out.write(np.zeros((480, 640, 3), dtype=np.uint8))
    out.release()

    label = predict_video(tmp, every_n=1)
    assert label in (0, 1)
    tmp.unlink()
