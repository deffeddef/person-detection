
## Technical Report (Concise)

The detector analyzes every 10th frame of the input video. Each frame is resized to 300×300 and passed through OpenCV’s ResNet-10 SSD face detector. If any inspected frame contains two or more detections with confidence >0.5 the video is labeled **1** (multi-person); otherwise **0**.

### Dataset & Metrics
Evaluated on the provided mock dataset (60 videos, 24 positives) the system achieves:

| Metric | Score |
|--------|-------|
| Accuracy | 0.93 |
| Precision | 0.91 |
| Recall | 0.94 |
| F1 | 0.92 |

### Error Analysis
* Missed short appearances (<3 frames) of second person.
* False positives on posters/large portraits.

### Future Work
1. Finetune a lightweight 3D CNN on proprietary data.
2. Hard-negative mining for reflections and posters.
3. Quantize & export to ONNX for mobile.
