"""
detect_from_image.py
---------------------
Run face mask detection on a single static image.

Usage:
    python detect_from_image.py --image path/to/image.jpg
    python detect_from_image.py --image path/to/image.jpg --output result.jpg
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description="Mask detection on a single image")
ap.add_argument("-i", "--image", required=True,
                help="Path to input image")
ap.add_argument("-f", "--face", default="face_detector",
                help="Path to face detector model directory")
ap.add_argument("-m", "--model", default=os.path.join("models", "mask_detector.model"),
                help="Path to trained mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="Minimum face detection confidence (default: 0.5)")
ap.add_argument("-o", "--output", type=str, default=None,
                help="Path to save annotated output image")
args = vars(ap.parse_args())

# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Loading face detector model...")
prototxt_path = os.path.join(args["face"], "deploy.prototxt")
weights_path = os.path.join(args["face"], "res10_300x300_ssd_iter_140000.caffemodel")

if not os.path.exists(prototxt_path) or not os.path.exists(weights_path):
    print("[ERROR] Face detector files not found. Run: python download_dataset.py")
    exit(1)

faceNet = cv2.dnn.readNet(prototxt_path, weights_path)

print("[INFO] Loading mask detector model...")
if not os.path.exists(args["model"]):
    print(f"[ERROR] Model not found at '{args['model']}'. Run: python train_mask_detector.py")
    exit(1)
maskNet = load_model(args["model"])

# ─────────────────────────────────────────────────────────────────────────────
# Load and process image
# ─────────────────────────────────────────────────────────────────────────────
print(f"[INFO] Processing image: {args['image']}")
image = cv2.imread(args["image"])
if image is None:
    print(f"[ERROR] Could not load image: {args['image']}")
    exit(1)

orig = image.copy()
(h, w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
faceNet.setInput(blob)
detections = faceNet.forward()

print(f"[INFO] Detected {detections.shape[2]} potential face regions.")
faces_detected = 0

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence < args["confidence"]:
        continue

    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    (startX, startY) = (max(0, startX), max(0, startY))
    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

    face = image[startY:endY, startX:endX]
    if face.size == 0:
        continue

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (224, 224))
    face_arr = img_to_array(face_rgb)
    face_arr = preprocess_input(face_arr)
    face_arr = np.expand_dims(face_arr, axis=0)

    (mask_prob, no_mask_prob) = maskNet.predict(face_arr)[0]
    label = "Mask" if mask_prob > no_mask_prob else "No Mask"
    prob = max(mask_prob, no_mask_prob) * 100
    color = (0, 200, 0) if label == "Mask" else (0, 0, 220)

    label_text = f"{label}: {prob:.1f}%"
    (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(image, (startX, startY - lh - 12), (startX + lw, startY), color, -1)
    cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    cv2.putText(image, label_text, (startX, startY - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    print(f"  Face {faces_detected + 1}: {label_text} (face confidence: {confidence * 100:.1f}%)")
    faces_detected += 1

if faces_detected == 0:
    print("[WARNING] No faces detected above confidence threshold.")

# ─────────────────────────────────────────────────────────────────────────────
# Show / save result
# ─────────────────────────────────────────────────────────────────────────────
if args["output"]:
    cv2.imwrite(args["output"], image)
    print(f"[INFO] Annotated image saved to: {args['output']}")
else:
    cv2.imshow("Mask Detection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("[DONE]")
