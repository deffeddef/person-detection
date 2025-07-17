
# Veriff Multi-Person Detector

Baseline reference implementation for detecting whether a verification video contains more than one person. The system samples frames, counts faces with OpenCV’s SSD face detector, and returns 0 (single person) or 1 (multiple people).

## Quick start
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/predict.py /path/to/video.mp4  # -> 0 or 1
```

## Batch evaluation
```bash
python src/evaluate.py --videos_dir ./videos --labels_file labels.txt
```

## Contents
* `src/` – core library and CLI scripts
* `tests/` – minimal unit tests
* `report.md` – short technical report with metrics
* `Dockerfile` – container recipe

## License
MIT
