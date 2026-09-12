"""
utils/alert_system.py
----------------------
Multi-level audio and visual alert system for drowsiness events.

Alert Levels:
  1 - CAUTION   : Eyes closing (EAR low, short duration)
  2 - WARNING   : Sustained eye closure or yawning
  3 - DANGER    : Prolonged closure + head drop (emergency)
"""

import threading
import time
import os
import sys


class AlertSystem:
    """
    Thread-safe alert manager with cooldown logic.
    Uses pygame for cross-platform audio playback.
    """

    LEVELS = {
        1: {"name": "CAUTION",  "color": (0, 200, 255),  "beeps": 1},
        2: {"name": "WARNING",  "color": (0, 140, 255),  "beeps": 2},
        3: {"name": "DANGER",   "color": (0, 0, 255),    "beeps": 4},
    }

    def __init__(self, cooldown_sec: float = 2.0):
        self.cooldown    = cooldown_sec
        self._last_alert = 0.0
        self._lock       = threading.Lock()
        self._audio_ok   = self._init_audio()
        self.current_level = 0

    # ── Audio ──────────────────────────────────────────────────────────────
    def _init_audio(self) -> bool:
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            return True
        except Exception:
            return False

    def _beep(self, freq: int = 880, duration_ms: int = 300):
        """Generate a synthetic beep using pygame."""
        if not self._audio_ok:
            return
        try:
            import pygame
            import numpy as np
            sample_rate = 44100
            n_samples   = int(sample_rate * duration_ms / 1000)
            t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)
            wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            # Stereo
            stereo = np.column_stack([wave, wave])
            sound  = pygame.sndarray.make_sound(stereo)
            sound.play()
            pygame.time.wait(duration_ms + 50)
        except Exception:
            pass

    # ── Public API ─────────────────────────────────────────────────────────
    def trigger(self, level: int):
        """Fire an alert at the given level (1/2/3) if not in cooldown."""
        now = time.time()
        with self._lock:
            if now - self._last_alert < self.cooldown:
                return
            self._last_alert = now
            self.current_level = level

        info = self.LEVELS.get(level, self.LEVELS[1])
        t = threading.Thread(target=self._play_alert,
                             args=(info["beeps"],), daemon=True)
        t.start()

    def _play_alert(self, beeps: int):
        freqs = [660, 880, 1100]
        for i in range(beeps):
            self._beep(freq=freqs[min(i, 2)], duration_ms=250)
            time.sleep(0.1)

    def clear(self):
        self.current_level = 0

    def get_alert_info(self, level: int) -> dict:
        return self.LEVELS.get(level, {"name": "OK", "color": (0, 200, 0), "beeps": 0})
