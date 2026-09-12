"""
download_dataset.py
-------------------
Downloads and organizes the face mask dataset from a public source.

Usage:
    python download_dataset.py

Output:
    dataset/with_mask/     - ~690 images of people wearing masks
    dataset/without_mask/  - ~686 images of people without masks
    face_detector/         - OpenCV DNN face detector model files
"""

import os
import sys
import zipfile
import urllib.request
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Dataset source: Prajna Bhandary's face mask detection dataset (GitHub)
# ─────────────────────────────────────────────────────────────────────────────
DATASET_URL = (
    "https://github.com/prajnasb/observations/archive/refs/heads/master.zip"
)

# OpenCV DNN face detector (Caffe ResNet10 SSD)
PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
WEIGHTS_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector"
    "_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FACE_DIR = os.path.join(BASE_DIR, "face_detector")


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: str, desc: str = "Downloading"):
    """Download a file with a tqdm progress bar."""
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)


def download_dataset():
    """Download and extract the face mask image dataset."""
    zip_path = os.path.join(BASE_DIR, "dataset_raw.zip")

    if not os.path.exists(zip_path):
        print("[INFO] Downloading face mask dataset (~45 MB)...")
        download_file(DATASET_URL, zip_path, "Face Mask Dataset")
    else:
        print("[INFO] Dataset zip already downloaded.")

    print("[INFO] Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(BASE_DIR)

    # Source path inside extracted archive
    src_base = os.path.join(BASE_DIR, "observations-master", "experiements",
                             "data")

    with_mask_src = os.path.join(src_base, "with_mask")
    without_mask_src = os.path.join(src_base, "without_mask")

    with_mask_dst = os.path.join(DATASET_DIR, "with_mask")
    without_mask_dst = os.path.join(DATASET_DIR, "without_mask")

    os.makedirs(with_mask_dst, exist_ok=True)
    os.makedirs(without_mask_dst, exist_ok=True)

    import shutil

    # Copy images
    for folder, src, dst in [
        ("with_mask", with_mask_src, with_mask_dst),
        ("without_mask", without_mask_src, without_mask_dst),
    ]:
        if os.path.exists(src):
            files = os.listdir(src)
            print(f"[INFO] Copying {len(files)} '{folder}' images...")
            for f in files:
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
        else:
            print(f"[WARNING] Source not found: {src}")

    print(f"[INFO] Dataset ready:")
    print(f"  with_mask:    {len(os.listdir(with_mask_dst))} images")
    print(f"  without_mask: {len(os.listdir(without_mask_dst))} images")


def download_face_detector():
    """Download the OpenCV DNN face detector model."""
    os.makedirs(FACE_DIR, exist_ok=True)

    proto = os.path.join(FACE_DIR, "deploy.prototxt")
    weights = os.path.join(FACE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(proto):
        print("[INFO] Downloading deploy.prototxt...")
        download_file(PROTO_URL, proto, "deploy.prototxt")
    else:
        print("[INFO] deploy.prototxt already present.")

    if not os.path.exists(weights):
        print("[INFO] Downloading face detector weights (~10 MB)...")
        download_file(WEIGHTS_URL, weights, "Face Detector Weights")
    else:
        print("[INFO] Face detector weights already present.")

    print("[INFO] Face detector ready.")


if __name__ == "__main__":
    print("=" * 60)
    print("  Face Mask Detection — Dataset Downloader")
    print("=" * 60)
    download_face_detector()
    download_dataset()
    print("\n[DONE] All downloads complete. You can now run:")
    print("  python train_mask_detector.py")
