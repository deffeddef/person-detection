
import cv2 as _cv2
from pathlib import Path
import urllib.request

MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector"
PROTO = "deploy.prototxt"
WEIGHTS = "res10_300x300_ssd_iter_140000.caffemodel"

class FaceDetector:
    """Lazy-loaded OpenCV DNN face detector."""
    def __init__(self):
        self._net = None

    @property
    def net(self):
        if self._net is None:
            self._net = self._load_net()
        return self._net

    def _download(self, filename):
        cache_dir = Path.home() / '.cache/veriff_faces'
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / filename
        if not dest.exists():
            print(f'Downloading {filename}…')
            urllib.request.urlretrieve(f'{MODEL_URL}/{filename}', dest)
        return str(dest)

    def _load_net(self):
        proto = self._download(PROTO)
        weights = self._download(WEIGHTS)
        net = _cv2.dnn.readNetFromCaffe(proto, weights)
        return net

    def detect(self, frame, conf_threshold=0.5):
        h, w = frame.shape[:2]
        blob = _cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()[0, 0]
        boxes = []
        for det in detections:
            confidence = float(det[2])
            if confidence >= conf_threshold:
                box = det[3:7] * [w, h, w, h]
                boxes.append(box.astype(int))
        return boxes

_detector = FaceDetector()

def get_detector():
    return _detector
