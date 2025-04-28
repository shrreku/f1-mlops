"""
Example script to run training on Kaggle environment using the ULB Credit Card Fraud dataset.

Steps in a Kaggle notebook:
1. Clone this repo:
   ```bash
   !git clone https://github.com/yourusername/f1-mlops.git
   %cd f1-mlops
   ```
2. Install dependencies:
   ```bash
   !pip install -r requirements.txt
   ```
3. Run this script:
   ```bash
   !python notebooks/train_on_kaggle.py
   ```
"""
import argparse
from src.train import main


def parse_args():
    parser = argparse.ArgumentParser(description='Train fraud detection on Kaggle')
    parser.add_argument(
        '--data-dir', type=str,
        default='/kaggle/input/mlg-ulb-creditcardfraud',
        help='Directory containing the Kaggle dataset'
    )
    parser.add_argument(
        '--artifact-dir', type=str,
        default='artifacts',
        help='Artifact output directory'
    )
    parser.add_argument(
        '--test-size', type=float,
        default=0.2,
        help='Test split ratio'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
