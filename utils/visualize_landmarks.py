"""
utils/visualize_landmarks.py
------------------------------
Standalone script to visualize all 468 MediaPipe Face Mesh landmarks
with labeled indices. Useful for understanding/debugging.

Usage:
    python utils/visualize_landmarks.py           # webcam
    python utils/visualize_landmarks.py --image face.jpg
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import mediapipe as mp

ap = argparse.ArgumentParser()
ap.add_argument("--image", default=None)
ap.add_argument("--source", default="0")
args = ap.parse_args()

mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Highlight indices of interest
HIGHLIGHT = {1: "NOSE", 152: "CHIN", 33: "R_EYE_L",
             263: "L_EYE_R", 61: "MOUTH_L", 291: "MOUTH_R"}

def process_frame(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0]
        mp_drawing.draw_landmarks(
            frame, lm,
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
        )
        # Annotate highlighted landmarks
        for idx, name in HIGHLIGHT.items():
            p = lm.landmark[idx]
            x, y = int(p.x * w), int(p.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(frame, f"{idx}:{name}", (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    return frame

if args.image:
    frame = cv2.imread(args.image)
    if frame is None:
        print("Cannot read image")
        sys.exit(1)
    out = process_frame(frame)
    cv2.imwrite("landmark_visualization.jpg", out)
    print("Saved: landmark_visualization.jpg")
else:
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = process_frame(frame)
        cv2.imshow("MediaPipe Landmarks (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
face_mesh.close()
