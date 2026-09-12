"""
utils/session_logger.py
------------------------
Logs all drowsiness events to CSV for post-session analysis.

Output: logs/session_<timestamp>.csv
Columns: timestamp, ear_left, ear_right, ear_avg, mar, pitch, yaw, roll,
         status, alert_level, event
"""

import csv
import os
import time
from datetime import datetime


class SessionLogger:
    """Append-mode CSV logger for drowsiness session data."""

    HEADERS = [
        "timestamp", "elapsed_sec",
        "ear_left", "ear_right", "ear_avg",
        "mar",
        "pitch", "yaw", "roll",
        "status", "alert_level", "event"
    ]

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"session_{ts}.csv")
        self._start   = time.time()
        self._file    = open(self.filepath, "w", newline="")
        self._writer  = csv.DictWriter(self._file, fieldnames=self.HEADERS)
        self._writer.writeheader()
        self._file.flush()
        print(f"[LOG] Session logging to: {self.filepath}")

    def log(self, ear_l: float, ear_r: float, mar: float,
            pitch: float, yaw: float, roll: float,
            status: str, alert_level: int, event: str = ""):
        elapsed = round(time.time() - self._start, 2)
        row = {
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "elapsed_sec": elapsed,
            "ear_left":    round(ear_l, 4),
            "ear_right":   round(ear_r, 4),
            "ear_avg":     round((ear_l + ear_r) / 2, 4),
            "mar":         round(mar, 4),
            "pitch":       round(pitch, 2),
            "yaw":         round(yaw, 2),
            "roll":        round(roll, 2),
            "status":      status,
            "alert_level": alert_level,
            "event":       event,
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.close()
        print(f"[LOG] Session saved: {self.filepath}")

    def get_path(self) -> str:
        return self.filepath
