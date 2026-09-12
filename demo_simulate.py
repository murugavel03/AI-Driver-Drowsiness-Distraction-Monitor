"""
demo_simulate.py
-----------------
Generates a synthetic session CSV log to demonstrate the analysis pipeline
WITHOUT needing a webcam. Useful for testing/grading environments.

Usage:
    python demo_simulate.py          # generates logs/session_demo.csv
    python demo_simulate.py --plot   # also runs analysis immediately
"""

import argparse
import os
import time
import math
import random
import csv
from datetime import datetime, timedelta


def simulate_ear(t: float, drowsy_intervals: list) -> float:
    """
    Simulate realistic EAR signal with blinks and drowsy dips.
    Normal open-eye EAR ≈ 0.30, blink ≈ 0.15, drowsy ≈ 0.18
    """
    base_ear = 0.30 + 0.02 * math.sin(0.3 * t)   # slow natural variation

    # Periodic blinks every ~4s
    blink_phase = t % 4.0
    if 0 < blink_phase < 0.15:
        base_ear = 0.14 + random.gauss(0, 0.01)

    # Drowsy intervals → prolonged low EAR
    for (start, end) in drowsy_intervals:
        if start <= t <= end:
            base_ear = 0.19 + 0.04 * math.sin(2 * t) + random.gauss(0, 0.01)
            break

    return max(0.05, base_ear + random.gauss(0, 0.005))


def simulate_mar(t: float, yawn_intervals: list) -> float:
    """Simulate MAR with periodic yawns."""
    mar = 0.20 + 0.05 * math.sin(0.1 * t) + random.gauss(0, 0.01)
    for (start, end) in yawn_intervals:
        if start <= t <= end:
            phase = (t - start) / max(end - start, 1)
            mar = 0.60 + 0.35 * math.sin(math.pi * phase) + random.gauss(0, 0.02)
            break
    return max(0.1, mar)


def simulate_head_pose(t: float, distract_intervals: list):
    """Simulate head pitch/yaw/roll with distraction events."""
    pitch = 5  * math.sin(0.05 * t) + random.gauss(0, 1)
    yaw   = 8  * math.sin(0.07 * t) + random.gauss(0, 1.5)
    roll  = 3  * math.sin(0.03 * t) + random.gauss(0, 0.5)

    for (start, end) in distract_intervals:
        if start <= t <= end:
            yaw   += 40 * math.sin(math.pi * (t - start) / max(end - start, 1))
            pitch += 15 * math.sin(math.pi * (t - start) / max(end - start, 1))
            break

    return pitch, yaw, roll


def determine_status(ear_avg: float, mar: float,
                     pitch: float, yaw: float,
                     consec_frames: int) -> tuple:
    EAR_THRESH = 0.25
    HEAD_PITCH  = 20.0
    HEAD_YAW    = 35.0

    head_distr = abs(yaw) > HEAD_YAW or abs(pitch) > HEAD_PITCH

    if consec_frames >= 60:
        return "DANGER", 3
    elif consec_frames >= 30:
        return "WARNING", 2
    elif consec_frames >= 15 or head_distr:
        return "CAUTION", 1
    else:
        return "ALERT", 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simulate a driver safety session")
    ap.add_argument("--duration", type=int, default=120,
                    help="Session duration in seconds (default: 120)")
    ap.add_argument("--fps",  type=float, default=10.0,
                    help="Simulated FPS (default: 10)")
    ap.add_argument("--plot", action="store_true",
                    help="Also run analyze_session.py after generating")
    args = ap.parse_args()

    os.makedirs("logs", exist_ok=True)
    out_path = "logs/session_demo.csv"

    # Scenario events
    drowsy_intervals   = [(30, 38), (70, 85), (100, 115)]
    yawn_intervals     = [(25, 28), (65, 69), (95, 99)]
    distract_intervals = [(45, 50), (90, 93)]

    dt_frame = 1.0 / args.fps
    T = args.duration
    N = int(T * args.fps)

    HEADERS = ["timestamp","elapsed_sec","ear_left","ear_right","ear_avg",
               "mar","pitch","yaw","roll","status","alert_level","event"]

    print(f"[SIM] Generating {N} frames ({T}s @ {args.fps} FPS)...")

    t0 = datetime.now()
    consec = 0
    blink_flag = False
    prev_ear_ok = True
    blink_events = []
    yawn_events  = []
    prev_yawn    = False

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()

        for i in range(N):
            t = i * dt_frame
            ts = (t0 + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            ear_l = simulate_ear(t, drowsy_intervals) + random.gauss(0, 0.005)
            ear_r = simulate_ear(t, drowsy_intervals) + random.gauss(0, 0.005)
            ear_avg = (ear_l + ear_r) / 2
            mar = simulate_mar(t, yawn_intervals)
            pitch, yaw, roll = simulate_head_pose(t, distract_intervals)

            ear_ok = ear_avg >= 0.25
            if not prev_ear_ok and ear_ok:
                event = "blink_detected"
            elif mar > 0.65 and not prev_yawn:
                event = "yawn_start"
                prev_yawn = True
            elif mar <= 0.65 and prev_yawn:
                event = "yawn_detected"
                prev_yawn = False
            else:
                event = ""

            prev_ear_ok = ear_ok

            if not ear_ok:
                consec += 1
            else:
                consec = 0

            status, alert_lvl = determine_status(ear_avg, mar, pitch, yaw, consec)

            writer.writerow({
                "timestamp":   ts,
                "elapsed_sec": round(t, 2),
                "ear_left":    round(ear_l, 4),
                "ear_right":   round(ear_r, 4),
                "ear_avg":     round(ear_avg, 4),
                "mar":         round(mar, 4),
                "pitch":       round(pitch, 2),
                "yaw":         round(yaw, 2),
                "roll":        round(roll, 2),
                "status":      status,
                "alert_level": alert_lvl,
                "event":       event,
            })

    print(f"[SIM] Done! Log saved to: {out_path}")

    if args.plot:
        import subprocess, sys
        print("[SIM] Running analysis...")
        subprocess.run([sys.executable, "analyze_session.py", "--log", out_path])
