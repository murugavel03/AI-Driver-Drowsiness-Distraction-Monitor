"""
evaluate.py
------------
Evaluate the trained mask detector model on a held-out test set.
Outputs accuracy, precision, recall, F1-score, and a confusion matrix.

Usage:
    python evaluate.py
    python evaluate.py --dataset dataset --model models/mask_detector.model
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from imutils import paths
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description="Evaluate face mask detector")
ap.add_argument("-d", "--dataset", default="dataset",
                help="Path to input dataset directory")
ap.add_argument("-m", "--model", default=os.path.join("models", "mask_detector.model"),
                help="Path to trained mask detector model")
ap.add_argument("-p", "--plot", default=os.path.join("plots", "confusion_matrix.png"),
                help="Path to save confusion matrix plot")
args = vars(ap.parse_args())

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ─────────────────────────────────────────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Loading images...")
image_paths = list(paths.list_images(args["dataset"]))

if len(image_paths) == 0:
    print("[ERROR] No images found. Run: python download_dataset.py")
    exit(1)

data, labels = [], []
for img_path in image_paths:
    label = img_path.split(os.sep)[-2]
    img = load_img(img_path, target_size=IMAGE_SIZE)
    img = img_to_array(img)
    img = preprocess_input(img)
    data.append(img)
    labels.append(label)

data = np.array(data, dtype="float32")
labels_raw = np.array(labels)

lb = LabelBinarizer()
labels_enc = lb.fit_transform(labels_raw)
labels_cat = to_categorical(labels_enc)

(_, testX, _, testY) = train_test_split(
    data, labels_cat, test_size=0.20, stratify=labels_cat, random_state=42
)

print(f"[INFO] Test set size: {len(testX)} images")
print(f"[INFO] Classes: {lb.classes_}")

# ─────────────────────────────────────────────────────────────────────────────
# Load model and predict
# ─────────────────────────────────────────────────────────────────────────────
print(f"[INFO] Loading model from '{args['model']}'...")
if not os.path.exists(args["model"]):
    print(f"[ERROR] Model not found. Run: python train_mask_detector.py")
    exit(1)

model = load_model(args["model"])

print("[INFO] Running predictions...")
predictions = model.predict(testX, batch_size=BATCH_SIZE)
pred_classes = np.argmax(predictions, axis=1)
true_classes = testY.argmax(axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
acc  = accuracy_score(true_classes, pred_classes) * 100
prec = precision_score(true_classes, pred_classes, average="weighted") * 100
rec  = recall_score(true_classes, pred_classes, average="weighted") * 100
f1   = f1_score(true_classes, pred_classes, average="weighted") * 100

print("\n" + "=" * 50)
print("  MODEL EVALUATION REPORT")
print("=" * 50)
print(f"  Accuracy  : {acc:.2f}%")
print(f"  Precision : {prec:.2f}%")
print(f"  Recall    : {rec:.2f}%")
print(f"  F1-Score  : {f1:.2f}%")
print("=" * 50 + "\n")

print("[INFO] Per-class Classification Report:")
print(classification_report(true_classes, pred_classes,
                              target_names=lb.classes_))

# ─────────────────────────────────────────────────────────────────────────────
# Confusion Matrix Plot
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(args["plot"]) or ".", exist_ok=True)

cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=lb.classes_, yticklabels=lb.classes_,
            linewidths=1, linecolor="gray")
plt.title("Confusion Matrix — Face Mask Detector", fontsize=14, fontweight="bold")
plt.ylabel("True Label", fontsize=12)
plt.xlabel("Predicted Label", fontsize=12)
plt.tight_layout()
plt.savefig(args["plot"], dpi=150)
print(f"[INFO] Confusion matrix saved to '{args['plot']}'")
print("[DONE] Evaluation complete!")
