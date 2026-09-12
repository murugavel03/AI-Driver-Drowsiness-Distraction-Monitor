"""
utils/face_metrics.py
----------------------
Core facial metric computations used by the drowsiness monitor.

Metrics implemented:
  - EAR  : Eye Aspect Ratio  (Soukupova & Cech, 2016)
  - MAR  : Mouth Aspect Ratio (yawn detection)
  - PUC  : Pupil Circularity  (additional eye feature)
  - Head Pose: pitch / yaw / roll via solvePnP
"""

import numpy as np
import cv2
from scipy.spatial import distance as dist


# ─── MediaPipe landmark indices (Face Mesh 468 points) ────────────────────────

# Left eye  (viewer's left)
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
# Right eye (viewer's right)
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# Mouth outer contour
MOUTH_IDX = [61, 291, 39, 181, 0, 17, 269, 405]

# Key landmarks for head pose (3D model points)
NOSE_TIP   = 1
CHIN       = 152
LEFT_EYE_L = 263
RIGHT_EYE_R= 33
LEFT_MOUTH = 287
RIGHT_MOUTH= 57

HEAD_POSE_LANDMARKS = [NOSE_TIP, CHIN, LEFT_EYE_L, RIGHT_EYE_R,
                       LEFT_MOUTH, RIGHT_MOUTH]

# 3D model reference points (mm, generic face model)
MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),    # Nose tip
    (0.0,  -330.0, -65.0),   # Chin
    (-225.0, 170.0, -135.0), # Left eye left corner
    (225.0,  170.0, -135.0), # Right eye right corner
    (-150.0,-150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0), # Right mouth corner
], dtype=np.float64)


def eye_aspect_ratio(landmarks, eye_indices, frame_w, frame_h):
    """
    Compute Eye Aspect Ratio (EAR).

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Returns float in [0, 1]. Closed eye ≈ 0.0, open eye ≈ 0.3
    """
    pts = np.array([
        [landmarks[i].x * frame_w, landmarks[i].y * frame_h]
        for i in eye_indices
    ], dtype=np.float64)

    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])

    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(landmarks, frame_w, frame_h):
    """
    Compute Mouth Aspect Ratio (MAR) for yawn detection.

    MAR = vertical opening / horizontal width
    High MAR (>0.6) indicates yawning.
    """
    pts = np.array([
        [landmarks[i].x * frame_w, landmarks[i].y * frame_h]
        for i in MOUTH_IDX
    ], dtype=np.float64)

    vertical   = dist.euclidean(pts[2], pts[6])
    horizontal = dist.euclidean(pts[0], pts[1])

    return vertical / (horizontal + 1e-6)


def head_pose_angles(landmarks, frame_w, frame_h):
    """
    Estimate head pose (pitch, yaw, roll) using solvePnP.

    Returns:
        pitch : up/down tilt  (positive = looking down)
        yaw   : left/right   (positive = looking right)
        roll  : head tilt    (positive = tilting right)
    All in degrees.
    """
    image_points = np.array([
        [landmarks[idx].x * frame_w, landmarks[idx].y * frame_h]
        for idx in HEAD_POSE_LANDMARKS
    ], dtype=np.float64)

    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    proj_matrix = np.hstack((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    pitch = float(euler_angles[0])
    yaw   = float(euler_angles[1])
    roll  = float(euler_angles[2])

    return pitch, yaw, roll
