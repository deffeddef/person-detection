# Experiments & Alternative Approaches

This document outlines some of the experimental choices made during the development of the person detection system and discusses potential alternative approaches.

## Model Selection

The current implementation uses a pre-trained SSD (Single Shot Detector) with a ResNet-10 backbone, provided by OpenCV. This model was chosen for several reasons:

*   **Performance:** It offers a good balance between speed and accuracy for face detection tasks.
*   **Simplicity:** It is easy to load and use with the OpenCV `dnn` module, requiring no external deep learning frameworks like TensorFlow or PyTorch.
*   **Lightweight:** The model is relatively small, making it suitable for deployment in resource-constrained environments.

## Alternative Approach: Haar Cascades

An alternative approach that was considered was using Haar Cascade classifiers, which are also available in OpenCV.

### What are Haar Cascades?

Haar Cascades are an older but still effective object detection method. They are essentially a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.

### Comparison

| Feature            | SSD (ResNet-10)                               | Haar Cascades                                     |
| ------------------ | --------------------------------------------- | ------------------------------------------------- |
| **Accuracy**       | Generally higher, more robust to variations in lighting and pose. | Lower, more prone to false positives.             |
| **Speed**          | Can be slower, especially without a GPU.      | Very fast on CPU.                                 |
| **Complexity**     | Simple to implement with `cv2.dnn`.           | Also simple to implement with `cv2.CascadeClassifier`. |
| **Robustness**     | Better at detecting faces at various angles and scales. | Less robust to non-frontal faces and occlusions. |

### Conclusion

While Haar Cascades are faster on CPU, the SSD-based detector was chosen as the primary model for this project due to its superior accuracy and robustness. Given that the goal is to reliably detect multiple people, minimizing false negatives (failing to detect a face) is more important than raw processing speed. The SSD model provides a better trade-off for this specific use case.

## Future Experiments

If further improvements were needed, the following experiments could be conducted:

*   **Fine-tuning:** The current model is used as-is. It could be fine-tuned on a custom dataset of verification videos to improve its performance in the specific domain.
*   **Other Architectures:** More modern architectures like YOLO (You Only Look Once) or EfficientDet could be explored. These might offer better performance but would require more complex dependencies (e.g., PyTorch, TensorFlow).
*   **Tracking:** Instead of just detecting faces in individual frames, a tracking algorithm could be used to track faces across frames. This would make the system more robust to temporary occlusions or detection failures.
