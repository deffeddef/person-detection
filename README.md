# Person Detection

A computer vision system to detect if a video contains more than one person. The system samples frames from a video, uses a Single Shot Detector (SSD) model with a ResNet-10 backbone to detect faces, and returns `1` if two or more faces are consistently found, otherwise `0`.

This project has been refactored to address feedback regarding reliability, documentation, and testing.

## Project Structure

*   `src/`: Core application source code for prediction and evaluation.
*   `models/`: Contains the pre-trained face detection model files.
*   `tests/`: Unit and integration tests.
*   `Dockerfile`: Recipe for building the production Docker image.
*   `report.md`: A brief technical report with performance metrics.
*   `experiments.md`: Documentation of experiments and alternative approaches.

## Local Setup

### Prerequisites

*   Python 3.10+
*   `pip` and `venv`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/person-detection.git
    cd person-detection
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

### Usage

**To predict on a single video:**

```bash
python -m src.predict path/to/your/video.mp4
```
The command will output `1` if multiple people are detected, and `0` otherwise.

**To evaluate a batch of labeled videos:**

The evaluation script takes a directory of videos and a labels file. The labels file should have one line per video, formatted as `<video_filename> <label>`.

```bash
python -m src.evaluate --videos_dir /path/to/videos --labels_file /path/to/labels.txt
```

## Docker Setup

### Prerequisites

*   Docker installed and running.

### Build the Image

Build the Docker image using the provided `Dockerfile`:

```bash
docker build -t person-detection .
```

### Run the Container

To run predictions, you need to mount a directory containing your videos into the container.

**To predict on a single video:**

```bash
docker run --rm -v /path/to/your/videos:/data person-detection --video /data/video.mp4
```
*Replace `/path/to/your/videos` with the absolute path to the directory containing your video files.*

**To get help on available arguments:**

```bash
docker run --rm person-detection --help
```

## Testing

This project uses `pytest` for testing.

To run the test suite:

```bash
python -m pytest
```
The tests include unit tests for individual functions and integration tests that are skipped if the required test videos are not present.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.