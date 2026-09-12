"""
detect_mask_video.py
---------------------
Real-time face mask detection using webcam or video file.

Usage:
    # Use webcam (default)
    python detect_mask_video.py

    # Use a video file
    python detect_mask_video.py --input path/to/video.mp4

    # Save output to file
    python detect_mask_video.py --output output_video.avi

Press 'q' to quit.
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
import time

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description="Real-time mask detection")
ap.add_argument("-f", "--face", default="face_detector",
                help="Path to face detector model directory")
ap.add_argument("-m", "--model", default=os.path.join("models", "mask_detector.model"),
                help="Path to trained mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="Minimum face detection confidence (default: 0.5)")
ap.add_argument("-i", "--input", type=str, default=None,
                help="Path to input video file (leave blank for webcam)")
ap.add_argument("-o", "--output", type=str, default=None,
                help="Path to output video file (optional)")
args = vars(ap.parse_args())


def detect_and_predict_mask(frame, faceNet, maskNet, confidence_thresh=0.5):
    """
    Detect faces in a frame and predict mask/no-mask for each face.

    Returns:
        locs   : list of bounding box (startX, startY, endX, endY) for each face
        preds  : list of (mask_prob, no_mask_prob) predictions
    """
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                  (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_thresh:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Clamp to frame boundaries
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)


def draw_prediction(frame, loc, pred):
    """Draw bounding box and label on the frame."""
    (startX, startY, endX, endY) = loc
    (mask, withoutMask) = pred

    label = "Mask" if mask > withoutMask else "No Mask"
    confidence = max(mask, withoutMask) * 100

    # Color: green for mask, red for no mask
    color = (0, 200, 0) if label == "Mask" else (0, 0, 220)

    label_text = f"{label}: {confidence:.1f}%"

    # Draw filled rectangle for label background
    (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (startX, startY - lh - 10), (startX + lw, startY), color, -1)

    # Draw bounding box
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Draw label text
    cv2.putText(frame, label_text, (startX, startY - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Loading face detector model...")
prototxt_path = os.path.join(args["face"], "deploy.prototxt")
weights_path = os.path.join(args["face"], "res10_300x300_ssd_iter_140000.caffemodel")

if not os.path.exists(prototxt_path) or not os.path.exists(weights_path):
    print("[ERROR] Face detector files not found.")
    print("        Please run: python download_dataset.py")
    exit(1)

faceNet = cv2.dnn.readNet(prototxt_path, weights_path)

print("[INFO] Loading mask detector model...")
if not os.path.exists(args["model"]):
    print(f"[ERROR] Mask model not found at '{args['model']}'.")
    print("        Please run: python train_mask_detector.py")
    exit(1)
maskNet = load_model(args["model"])

# ─────────────────────────────────────────────────────────────────────────────
# Video source
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Starting video stream...")
if args["input"] is None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    source_type = "webcam"
else:
    vs = cv2.VideoCapture(args["input"])
    source_type = "file"

writer = None

print("[INFO] Press 'q' to quit.")
while True:
    frame = vs.read()
    frame = frame[1] if source_type == "file" else frame

    if frame is None:
        print("[INFO] End of video stream.")
        break

    frame = imutils.resize(frame, width=800)

    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet,
                                             args["confidence"])

    for (loc, pred) in zip(locs, preds):
        draw_prediction(frame, loc, pred)

    # FPS counter
    cv2.putText(frame, "Press 'q' to quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Save output video
    if args["output"] is not None:
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 20,
                                     (frame.shape[1], frame.shape[0]), True)
        writer.write(frame)

    cv2.imshow("Face Mask Detector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# Cleanup
if writer is not None:
    writer.release()
if source_type == "webcam":
    vs.stop()
else:
    vs.release()
cv2.destroyAllWindows()
print("[DONE] Detection complete.")
