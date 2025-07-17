
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from .utils import predict_video

def load_labels(labels_file):
    with open(labels_file) as f:
        return {Path(line.split()[0]).name: int(line.split()[1]) for line in f}

def main():
    p = argparse.ArgumentParser(description='Evaluate detector on labelled videos')
    p.add_argument('--videos_dir', type=Path, required=True)
    p.add_argument('--labels_file', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()

    labels = load_labels(args.labels_file)
    videos = list(args.videos_dir.glob('*.mp4'))

    y_true, y_pred = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for pred in tqdm(ex.map(predict_video, videos)):
            y_pred.append(pred)
    y_true = [labels[v.name] for v in videos]

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f'Accuracy: {acc:.2f}
Precision: {prec:.2f}
Recall: {rec:.2f}
F1: {f1:.2f}')

if __name__ == '__main__':
    main()
