
"""
This script evaluates the performance of the person detector on a labeled dataset.

It takes a directory of videos and a file with corresponding labels, runs the
prediction on each video, and then computes and prints common classification
metrics (Accuracy, Precision, Recall, F1-score).
"""
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from .utils import predict_video

def load_labels(labels_file: Path) -> dict[str, int]:
    """
    Loads labels from a text file.

    The file is expected to have one label per line, in the format:
    <video_filename> <label>

    Args:
        labels_file: Path to the labels file.

    Returns:
        A dictionary mapping video filenames to their integer labels.
    """
    with open(labels_file) as f:
        return {Path(line.split()[0]).name: int(line.split()[1]) for line in f}

def main():
    """
    Main function to run the evaluation.
    """
    parser = argparse.ArgumentParser(description='Evaluate detector on labelled videos')
    parser.add_argument('--videos_dir', type=Path, required=True, help='Directory containing the video files.')
    parser.add_argument('--labels_file', type=Path, required=True, help='File containing the video labels.')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads to use for processing videos.')
    args = parser.parse_args()

    # Load the ground truth labels
    labels = load_labels(args.labels_file)
    videos = sorted(list(args.videos_dir.glob('*.mp4'))) # Sort to ensure order
    video_names = [v.name for v in videos]

    # Ensure all videos in the directory have a corresponding label
    if not set(video_names).issubset(set(labels.keys())):
        unlabeled_videos = set(video_names) - set(labels.keys())
        raise ValueError(f"Missing labels for the following videos: {unlabeled_videos}")

    y_true = [labels[name] for name in video_names]
    y_pred = []

    # Use a thread pool to process videos in parallel for faster evaluation
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        # The map function will preserve the order of the videos
        results = ex.map(predict_video, videos)
        # Wrap with tqdm to show a progress bar
        for pred in tqdm(results, total=len(videos), desc="Evaluating videos"):
            y_pred.append(pred)

    # Calculate and print the evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    print("\n--- Evaluation Results ---")
    print(f"Accuracy:  {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall:    {rec:.2f}")
    print(f"F1-score:  {f1:.2f}")
    print("--------------------------")

if __name__ == '__main__':
    main()
