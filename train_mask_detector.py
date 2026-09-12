"""
train_mask_detector.py
-----------------------
Trains a face mask detector using Transfer Learning on MobileNetV2.

Usage:
    python train_mask_detector.py [--dataset dataset] [--model models/mask_detector.model]
                                  [--plot plots/mask_training_plot.png] [--epochs 20]

Output:
    - models/mask_detector.model     : Trained Keras model
    - plots/mask_training_plot.png   : Training accuracy/loss curves
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
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    AveragePooling2D, Dropout, Flatten, Dense, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description="Train face mask detector")
ap.add_argument("-d", "--dataset", default="dataset",
                help="Path to input dataset directory")
ap.add_argument("-m", "--model", default=os.path.join("models", "mask_detector.model"),
                help="Path to output trained model")
ap.add_argument("-p", "--plot", default=os.path.join("plots", "mask_training_plot.png"),
                help="Path to output loss/accuracy plot")
ap.add_argument("-e", "--epochs", type=int, default=20,
                help="Number of training epochs (default: 20)")
args = vars(ap.parse_args())

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
INIT_LR = 1e-4
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load images and labels
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Loading images from dataset...")
image_paths = list(paths.list_images(args["dataset"]))

if len(image_paths) == 0:
    print("[ERROR] No images found. Please run: python download_dataset.py")
    exit(1)

data = []
labels = []

for img_path in image_paths:
    # Extract class label from directory name
    label = img_path.split(os.sep)[-2]

    img = load_img(img_path, target_size=IMAGE_SIZE)
    img = img_to_array(img)
    img = preprocess_input(img)

    data.append(img)
    labels.append(label)

print(f"[INFO] Loaded {len(data)} images.")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Encode labels and split data
# ─────────────────────────────────────────────────────────────────────────────
data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=42
)

# Data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Build model (MobileNetV2 + custom head)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Building model with MobileNetV2 backbone...")

baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

# Freeze base model layers
for layer in baseModel.layers:
    layer.trainable = False

# Build custom classifier head
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Compile and train
# ─────────────────────────────────────────────────────────────────────────────
print(f"[INFO] Compiling model (LR={INIT_LR}, epochs={args['epochs']})...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / args["epochs"])
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Training model...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BATCH_SIZE,
    epochs=args["epochs"]
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Evaluate
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Evaluating network...")
predIdxs = model.predict(testX, batch_size=BATCH_SIZE)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(
    testY.argmax(axis=1),
    predIdxs,
    target_names=lb.classes_
))

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Save model
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(args["model"]) or ".", exist_ok=True)
print(f"[INFO] Saving model to '{args['model']}'...")
model.save(args["model"])

# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Plot training curves
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(args["plot"]) or ".", exist_ok=True)
N = args["epochs"]
plt.style.use("ggplot")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.set_title("Training vs Validation Loss")
ax1.plot(np.arange(0, N), H.history["loss"], label="train_loss", color="#e74c3c")
ax1.plot(np.arange(0, N), H.history["val_loss"], label="val_loss", color="#c0392b", linestyle="--")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.set_title("Training vs Validation Accuracy")
ax2.plot(np.arange(0, N), H.history["accuracy"], label="train_acc", color="#2ecc71")
ax2.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc", color="#27ae60", linestyle="--")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig(args["plot"], dpi=150)
print(f"[INFO] Training plot saved to '{args['plot']}'")
print("[DONE] Training complete!")
