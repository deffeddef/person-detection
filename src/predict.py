
import argparse
from pathlib import Path
from .utils import predict_video

def main():
    parser = argparse.ArgumentParser(description='Predict if video contains >1 person.')
    parser.add_argument('video', type=Path, help='Path to input video')
    parser.add_argument('--every', type=int, default=10, help='Sample every Nth frame')
    args = parser.parse_args()

    label = predict_video(args.video, every_n=args.every)
    print(label)

if __name__ == '__main__':
    main()
