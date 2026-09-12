"""
monitor.py
-----------
AI-Powered Real-Time Driver Drowsiness & Distraction Monitor.

Detects:
  ✦ Eye drowsiness      — Eye Aspect Ratio (EAR) + blink rate
  ✦ Yawning             — Mouth Aspect Ratio (MAR)
  ✦ Head pose deviation — pitch / yaw / roll via solvePnP
  ✦ Microsleep          — sustained eye closure (> CONSEC_FRAMES_DANGER)

Uses: MediaPipe Face Mesh (468 3D landmarks) — no dataset/training needed.

Usage:
    python monitor.py                          # webcam (default)
    python monitor.py --source video.mp4       # video file
    python monitor.py --source 0 --no-log      # webcam, disable CSV log
    python monitor.py --ear 0.25 --mar 0.65    # custom thresholds
    python monitor.py --calibrate              # run auto-calibration first

Press:
    q  — quit
    s  — save current frame screenshot
    r  — reset session counters
    c  — toggle landmark overlay
"""

import argparse
import os
import sys
import time
import math
import datetime
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import mediapipe as mp

# Local utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.face_metrics  import (eye_aspect_ratio, mouth_aspect_ratio,
                                  head_pose_angles,
                                  LEFT_EYE_IDX, RIGHT_EYE_IDX)
from utils.alert_system  import AlertSystem
from utils.session_logger import SessionLogger


# ─────────────────────────────────────────────────────────────────────────────
# Default Thresholds
# ─────────────────────────────────────────────────────────────────────────────
EAR_THRESHOLD          = 0.25   # Below → eye closing
MAR_THRESHOLD          = 0.65   # Above → yawning
HEAD_PITCH_THRESHOLD   = 20.0   # degrees down → nodding off
HEAD_YAW_THRESHOLD     = 35.0   # degrees left/right → distraction

CONSEC_FRAMES_CAUTION  = 15     # ~0.5s   → caution alert
CONSEC_FRAMES_WARNING  = 30     # ~1.0s   → warning alert
CONSEC_FRAMES_DANGER   = 60     # ~2.0s   → danger (microsleep)

YAWN_CONSEC_FRAMES     = 20     # sustained yawn threshold


# ─────────────────────────────────────────────────────────────────────────────
# HUD Drawing Helpers
# ─────────────────────────────────────────────────────────────────────────────
FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

STATUS_COLORS = {
    "ALERT":       (0, 220, 0),
    "CAUTION":     (0, 200, 255),
    "WARNING":     (0, 130, 255),
    "DANGER":      (0, 0, 255),
    "NO FACE":     (120, 120, 120),
}


def draw_filled_rect(img, x1, y1, x2, y2, color, alpha=0.55):
    """Draw a semi-transparent filled rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_metric_bar(img, x, y, label, value, vmin, vmax, threshold,
                    bar_w=150, bar_h=16, color_ok=(0,200,0), color_bad=(0,0,255)):
    """Draw a labelled metric progress bar with threshold marker."""
    pct = np.clip((value - vmin) / (vmax - vmin + 1e-6), 0, 1)
    tpct= np.clip((threshold - vmin) / (vmax - vmin + 1e-6), 0, 1)

    color = color_bad if value >= threshold else color_ok

    # Background
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), -1)
    # Fill
    fill_w = int(bar_w * pct)
    cv2.rectangle(img, (x, y), (x + fill_w, y + bar_h), color, -1)
    # Threshold line
    tx = x + int(bar_w * tpct)
    cv2.line(img, (tx, y - 2), (tx, y + bar_h + 2), (255, 255, 0), 2)
    # Border
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (180, 180, 180), 1)
    # Label
    cv2.putText(img, f"{label}: {value:.3f}", (x, y - 4),
                FONT, 0.42, (220, 220, 220), 1)


def draw_head_compass(img, cx, cy, r, pitch, yaw):
    """Draw a mini compass showing head orientation."""
    # Outer circle
    cv2.circle(img, (cx, cy), r, (60, 60, 60), -1)
    cv2.circle(img, (cx, cy), r, (140, 140, 140), 2)

    # Cross-hairs
    cv2.line(img, (cx - r, cy), (cx + r, cy), (80, 80, 80), 1)
    cv2.line(img, (cx, cy - r), (cx, cy + r), (80, 80, 80), 1)

    # Dot position
    dx = int(np.clip(yaw   / HEAD_YAW_THRESHOLD,   -1, 1) * (r - 6))
    dy = int(np.clip(pitch / HEAD_PITCH_THRESHOLD, -1, 1) * (r - 6))
    dot_color = (0, 0, 230) if (abs(yaw) > HEAD_YAW_THRESHOLD or
                                 abs(pitch) > HEAD_PITCH_THRESHOLD) else (0, 220, 0)
    cv2.circle(img, (cx + dx, cy + dy), 6, dot_color, -1)
    cv2.circle(img, (cx + dx, cy + dy), 6, (255, 255, 255), 1)
    cv2.putText(img, "HEAD", (cx - 18, cy + r + 16), FONT, 0.38, (180, 180, 180), 1)


def draw_hud(img, metrics: dict, session: dict, show_landmarks: bool):
    """Render the full HUD overlay onto img in place."""
    h, w = img.shape[:2]

    status     = metrics["status"]
    status_col = STATUS_COLORS.get(status, (200, 200, 200))

    # ── Top status banner ─────────────────────────────────────────────────
    banner_h = 52
    draw_filled_rect(img, 0, 0, w, banner_h, (20, 20, 20), alpha=0.75)

    # Status pill
    pill_text = f"  {status}  "
    (pw, ph), _ = cv2.getTextSize(pill_text, FONT_BOLD, 0.9, 2)
    px = (w - pw) // 2
    draw_filled_rect(img, px - 10, 8, px + pw + 10, banner_h - 8, status_col, alpha=0.9)
    cv2.putText(img, pill_text, (px, banner_h - 14), FONT_BOLD, 0.9, (255, 255, 255), 2)

    # Blink count & yawn count (top-right)
    info_x = w - 220
    cv2.putText(img, f"Blinks: {session['blink_count']}",
                (info_x, 22), FONT, 0.55, (200, 200, 200), 1)
    cv2.putText(img, f"Yawns:  {session['yawn_count']}",
                (info_x, 44), FONT, 0.55, (200, 200, 200), 1)

    # Elapsed time (top-left)
    elapsed = int(time.time() - session["start_time"])
    mins, secs = divmod(elapsed, 60)
    cv2.putText(img, f"Session: {mins:02d}:{secs:02d}",
                (10, 22), FONT, 0.55, (180, 180, 180), 1)
    cv2.putText(img, f"FPS: {metrics['fps']:.1f}",
                (10, 44), FONT, 0.55, (150, 150, 150), 1)

    # ── Left sidebar — metric bars ─────────────────────────────────────────
    sb_x = 10
    sb_y_start = banner_h + 15

    draw_filled_rect(img, 0, banner_h, 200, h, (15, 15, 15), alpha=0.65)

    # EAR bars
    cv2.putText(img, "EYE METRICS", (sb_x, sb_y_start + 12),
                FONT, 0.45, (160, 160, 160), 1)
    draw_metric_bar(img, sb_x, sb_y_start + 22,
                    "L-EAR", metrics["ear_left"],
                    0.0, 0.45, EAR_THRESHOLD,
                    bar_w=175, color_ok=(0, 200, 80), color_bad=(0, 60, 220))
    draw_metric_bar(img, sb_x, sb_y_start + 58,
                    "R-EAR", metrics["ear_right"],
                    0.0, 0.45, EAR_THRESHOLD,
                    bar_w=175, color_ok=(0, 200, 80), color_bad=(0, 60, 220))
    draw_metric_bar(img, sb_x, sb_y_start + 94,
                    "AVG-EAR", (metrics["ear_left"] + metrics["ear_right"]) / 2,
                    0.0, 0.45, EAR_THRESHOLD,
                    bar_w=175, color_ok=(0, 180, 100), color_bad=(0, 30, 240))

    # MAR bar
    cv2.putText(img, "MOUTH", (sb_x, sb_y_start + 135),
                FONT, 0.45, (160, 160, 160), 1)
    draw_metric_bar(img, sb_x, sb_y_start + 145,
                    "MAR", metrics["mar"],
                    0.0, 1.2, MAR_THRESHOLD,
                    bar_w=175, color_ok=(0, 200, 80), color_bad=(30, 100, 255))

    # Head pose compass
    compass_cx = 95
    compass_cy = sb_y_start + 255
    draw_head_compass(img, compass_cx, compass_cy, 55,
                      metrics["pitch"], metrics["yaw"])

    cv2.putText(img, f"P:{metrics['pitch']:+.1f}", (sb_x, sb_y_start + 320),
                FONT, 0.40, (180, 180, 180), 1)
    cv2.putText(img, f"Y:{metrics['yaw']:+.1f}",
                (sb_x + 65, sb_y_start + 320), FONT, 0.40, (180, 180, 180), 1)
    cv2.putText(img, f"R:{metrics['roll']:+.1f}",
                (sb_x + 130, sb_y_start + 320), FONT, 0.40, (180, 180, 180), 1)

    # ── Bottom alert bar ──────────────────────────────────────────────────
    if metrics["alert_level"] > 0:
        al   = metrics["alert_level"]
        msg  = ["", "⚠ Eyes Closing — Stay Alert!",
                "⚠ WARNING: Drowsiness Detected!",
                "🚨 DANGER: MICROSLEEP DETECTED!"][al]
        bcol = [(0,0,0), (0, 160, 220), (0, 80, 255), (0, 0, 200)][al]
        draw_filled_rect(img, 0, h - 50, w, h, bcol, alpha=0.85)
        (tw, _), _ = cv2.getTextSize(msg, FONT_BOLD, 0.85, 2)
        cv2.putText(img, msg, ((w - tw) // 2, h - 14),
                    FONT_BOLD, 0.85, (255, 255, 255), 2)

    # ── Closure counter bar ───────────────────────────────────────────────
    if metrics["consec_frames"] > 0:
        pct = min(metrics["consec_frames"] / CONSEC_FRAMES_DANGER, 1.0)
        bar_y = h - 56
        bar_w = int((w - 210) * pct)
        bar_col = (0, int(200 * (1 - pct)), int(255 * pct))
        cv2.rectangle(img, (205, bar_y), (205 + bar_w, bar_y + 6), bar_col, -1)
        cv2.rectangle(img, (205, bar_y), (w, bar_y + 6), (60, 60, 60), 1)

    # ── Timestamp ─────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    cv2.putText(img, ts, (10, h - 8), FONT, 0.45, (100, 100, 100), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────
def calibrate(cap, face_mesh) -> float:
    """
    Sample EAR for 3 seconds while driver keeps eyes open.
    Returns recommended EAR threshold = mean_EAR * 0.75
    """
    print("[CALIBRATE] Keep eyes OPEN and look at the camera for 3 seconds...")
    ears = []
    start = time.time()
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    while time.time() - start < 3.0:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            el = eye_aspect_ratio(lm, LEFT_EYE_IDX,  w, h)
            er = eye_aspect_ratio(lm, RIGHT_EYE_IDX, w, h)
            ears.append((el + er) / 2)

        remaining = 3.0 - (time.time() - start)
        cv2.putText(frame, f"Calibrating... {remaining:.1f}s",
                    (30, 50), FONT_BOLD, 1.0, (0, 200, 255), 2)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyWindow("Calibration")

    if ears:
        mean_ear = np.mean(ears)
        threshold = round(mean_ear * 0.75, 3)
        print(f"[CALIBRATE] Mean open EAR = {mean_ear:.4f} → threshold = {threshold}")
        return threshold
    return EAR_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# Main Monitor
# ─────────────────────────────────────────────────────────────────────────────
def run_monitor(args):
    global EAR_THRESHOLD

    # ── MediaPipe Face Mesh ────────────────────────────────────────────────
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,       # enables iris landmarks (468+10)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # ── Video source ───────────────────────────────────────────────────────
    source = 0 if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(int(source) if isinstance(source, str)
                           and source.isdigit() else source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {args.source}")
        sys.exit(1)

    # ── Calibration ────────────────────────────────────────────────────────
    if args.calibrate:
        EAR_THRESHOLD = calibrate(cap, face_mesh)

    # Apply CLI threshold overrides
    ear_thresh = args.ear if args.ear else EAR_THRESHOLD
    mar_thresh = args.mar if args.mar else MAR_THRESHOLD

    # ── Logger & Alert ────────────────────────────────────────────────────
    logger = SessionLogger() if not args.no_log else None
    alerter = AlertSystem(cooldown_sec=2.0)

    # ── Session state ──────────────────────────────────────────────────────
    session = {
        "start_time":   time.time(),
        "blink_count":  0,
        "yawn_count":   0,
        "consec_frames": 0,      # consecutive low-EAR frames
        "yawn_frames":   0,
        "prev_ear_ok":   True,   # for blink edge detection
        "screenshots":   0,
    }

    show_landmarks = False

    # FPS tracking
    fps_buf  = []
    prev_t   = time.time()

    print("[INFO] Monitor running. Press 'q' to quit, 's' to screenshot, "
          "'r' to reset, 'c' to toggle landmarks.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        frame = cv2.flip(frame, 1)    # mirror for webcam
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # FPS
        now   = time.time()
        dt    = now - prev_t
        prev_t = now
        fps_buf.append(1.0 / max(dt, 1e-6))
        if len(fps_buf) > 30:
            fps_buf.pop(0)
        fps = np.mean(fps_buf)

        # ── Face Mesh inference ────────────────────────────────────────────
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        # ── Defaults (no face) ─────────────────────────────────────────────
        metrics = {
            "ear_left": 0.0, "ear_right": 0.0,
            "mar": 0.0,
            "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
            "status": "NO FACE",
            "alert_level": 0,
            "consec_frames": session["consec_frames"],
            "fps": fps,
        }

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # ── Draw landmarks ─────────────────────────────────────────────
            if show_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.multi_face_landmarks[0],
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()
                )

            # ── Compute metrics ────────────────────────────────────────────
            ear_l = eye_aspect_ratio(lm, LEFT_EYE_IDX,  w, h)
            ear_r = eye_aspect_ratio(lm, RIGHT_EYE_IDX, w, h)
            ear   = (ear_l + ear_r) / 2.0
            mar   = mouth_aspect_ratio(lm, w, h)
            pitch, yaw, roll = head_pose_angles(lm, w, h)

            metrics.update({
                "ear_left": ear_l, "ear_right": ear_r,
                "mar": mar,
                "pitch": pitch, "yaw": yaw, "roll": roll,
            })

            # ── Blink detection (rising edge of EAR) ─────────────────────
            ear_ok = ear >= ear_thresh
            if not session["prev_ear_ok"] and ear_ok:
                session["blink_count"] += 1
                if logger:
                    logger.log(ear_l, ear_r, mar, pitch, yaw, roll,
                               "BLINK", 0, "blink_detected")
            session["prev_ear_ok"] = ear_ok

            # ── Yawn detection ─────────────────────────────────────────────
            if mar > mar_thresh:
                session["yawn_frames"] += 1
            else:
                if session["yawn_frames"] >= YAWN_CONSEC_FRAMES:
                    session["yawn_count"] += 1
                    if logger:
                        logger.log(ear_l, ear_r, mar, pitch, yaw, roll,
                                   "YAWN", 1, "yawn_detected")
                session["yawn_frames"] = 0

            # ── Drowsiness level ─────────────────────────────────────────
            head_distracted = (abs(yaw) > HEAD_YAW_THRESHOLD or
                               abs(pitch) > HEAD_PITCH_THRESHOLD)

            if ear < ear_thresh:
                session["consec_frames"] += 1
            else:
                session["consec_frames"] = 0

            cf = session["consec_frames"]

            if cf >= CONSEC_FRAMES_DANGER:
                status = "DANGER"
                alert_lvl = 3
                alerter.trigger(3)
            elif cf >= CONSEC_FRAMES_WARNING:
                status = "WARNING"
                alert_lvl = 2
                alerter.trigger(2)
            elif cf >= CONSEC_FRAMES_CAUTION or head_distracted:
                status = "CAUTION"
                alert_lvl = 1
                alerter.trigger(1)
            else:
                status = "ALERT"
                alert_lvl = 0
                alerter.clear()

            metrics.update({
                "status": status,
                "alert_level": alert_lvl,
                "consec_frames": cf,
            })

            # ── Periodic logging ───────────────────────────────────────────
            if logger and int(elapsed := time.time() - session["start_time"]) % 1 == 0:
                logger.log(ear_l, ear_r, mar, pitch, yaw, roll,
                           status, alert_lvl)

        # ── Draw HUD ──────────────────────────────────────────────────────
        draw_hud(frame, metrics, session, show_landmarks)

        cv2.imshow("AI Driver Safety Monitor", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            os.makedirs("screenshots", exist_ok=True)
            fname = f"screenshots/capture_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            session["screenshots"] += 1
            print(f"[SNAP] Saved: {fname}")
        elif key == ord("r"):
            session["blink_count"]  = 0
            session["yawn_count"]   = 0
            session["consec_frames"] = 0
            session["start_time"]   = time.time()
            print("[RESET] Session counters reset.")
        elif key == ord("c"):
            show_landmarks = not show_landmarks

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()

    if logger:
        logger.close()

    print(f"\n{'='*50}")
    print(f"  SESSION SUMMARY")
    print(f"{'='*50}")
    elapsed = int(time.time() - session["start_time"])
    mins, secs = divmod(elapsed, 60)
    print(f"  Duration   : {mins:02d}:{secs:02d}")
    print(f"  Blinks     : {session['blink_count']}")
    print(f"  Yawns      : {session['yawn_count']}")
    print(f"  Screenshots: {session['screenshots']}")
    if logger:
        print(f"  Log saved  : {logger.get_path()}")
    print(f"{'='*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="AI Driver Drowsiness & Distraction Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python monitor.py                      # webcam
  python monitor.py --source video.mp4   # video file
  python monitor.py --calibrate          # auto-calibrate EAR threshold
  python monitor.py --ear 0.22 --mar 0.70
  python monitor.py --no-log             # disable CSV logging
        """
    )
    ap.add_argument("--source",    default="0",
                    help="Video source: 0 (webcam), 1, or path to video file")
    ap.add_argument("--ear",       type=float, default=None,
                    help=f"EAR threshold (default: {EAR_THRESHOLD})")
    ap.add_argument("--mar",       type=float, default=None,
                    help=f"MAR threshold (default: {MAR_THRESHOLD})")
    ap.add_argument("--no-log",    action="store_true",
                    help="Disable CSV session logging")
    ap.add_argument("--calibrate", action="store_true",
                    help="Run EAR auto-calibration before monitoring")

    args = ap.parse_args()
    run_monitor(args)
