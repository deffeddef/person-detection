import cv2 as _cv2
from pathlib import Path

# Define paths to the model files
MODEL_DIR = Path(__file__).parent.parent / 'models'
PROTO = str(MODEL_DIR / "deploy.prototxt")
WEIGHTS = str(MODEL_DIR / "res10_300x300_ssd_iter_140000_fp16.caffemodel")

class FaceDetector:
    """Lazy-loaded OpenCV DNN face detector."""
    def __init__(self):
        self._net = None

    @property
    def net(self):
        if self._net is None:
            self._net = self._load_net()
        return self._net

    def _load_net(self):
        """Loads the Caffe model from local files."""
        net = _cv2.dnn.readNetFromCaffe(PROTO, WEIGHTS)
        return net

    def detect(self, frame, conf_threshold=0.5):
        """
        Detects faces in a single frame.

        Args:
            frame: The input image frame.
            conf_threshold: The confidence threshold to filter weak detections.

        Returns:
            A list of bounding boxes for detected faces.
        """
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
    """Returns a singleton instance of the FaceDetector."""
    return _detector