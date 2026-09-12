# 🚗 AI Driver Drowsiness & Distraction Monitor

> **Real-time Computer Vision system** that monitors driver alertness using **MediaPipe Face Mesh** (468 3D landmarks), detecting drowsiness, yawning, and head pose deviation — with multi-level audio alerts and session analytics.

---

## 📌 Project Overview

| Attribute       | Details                                               |
|----------------|-------------------------------------------------------|
| **Domain**      | Computer Vision / Real-Time Systems                   |
| **Language**    | Python 3.8+                                           |
| **CV Library**  | MediaPipe (Google) + OpenCV 4.8                       |
| **Detection**   | Eye Aspect Ratio, Mouth Aspect Ratio, Head Pose (PnP) |
| **Alert System**| Multi-level audio alerts (pygame)                     |
| **Logging**     | Per-frame CSV logging → post-session analytics        |
| **No Training** | Uses pre-trained MediaPipe Face Mesh (instant setup)  |

---

## 🧠 What It Detects

| Event | Method | Threshold |
|-------|--------|-----------|
| 👁️ Eye closing / Drowsiness | Eye Aspect Ratio (EAR) | EAR < 0.25 for 0.5s |
| 😴 Microsleep | Consecutive low-EAR frames | > 2 seconds |
| 😮 Yawning | Mouth Aspect Ratio (MAR) | MAR > 0.65 |
| 🔄 Head nodding down | Head pitch angle | > 20° |
| ↔️ Distracted (looking away) | Head yaw angle | > 35° |

### Alert Levels
```
Level 1 — CAUTION  : Eyes beginning to close (yellow)
Level 2 — WARNING  : Sustained drowsiness (orange)
Level 3 — DANGER   : Microsleep detected — emergency (red)
```

---

## 📁 Project Structure

```
driver-drowsiness-monitor/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
│
├── monitor.py                      # 🔑 Main real-time detection (run this)
├── analyze_session.py              # Post-session analytics & charts
├── demo_simulate.py                # Demo without webcam (generates fake data)
│
├── utils/
│   ├── __init__.py
│   ├── face_metrics.py             # EAR, MAR, head pose computations
│   ├── alert_system.py             # Multi-level audio alert manager
│   ├── session_logger.py           # CSV session logging
│   └── visualize_landmarks.py      # MediaPipe landmark explorer
│
├── logs/                           # Auto-created: session CSV logs
├── plots/                          # Auto-created: analysis charts
└── screenshots/                    # Auto-created: saved frames
```

---

## ⚙️ Environment Setup

### Prerequisites
- Python **3.8–3.11** (MediaPipe requires ≤ 3.11)
- Webcam (or a video file for testing)
- ~500 MB disk space

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/driver-drowsiness-monitor.git
cd driver-drowsiness-monitor
```

### Step 2 — Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ If MediaPipe fails to install, ensure Python ≤ 3.11:
> ```bash
> python --version  # must be 3.8 – 3.11
> ```

---

## 🚀 Running the Project

### ▶️ Option A: Real-Time Webcam Monitor

```bash
python monitor.py
```

### ▶️ Option B: With a Video File

```bash
python monitor.py --source path/to/driving_video.mp4
```

### ▶️ Option C: Auto-Calibrate EAR Threshold First

```bash
python monitor.py --calibrate
```
Stare at the camera for 3 seconds; it auto-sets your personal EAR threshold.

### ▶️ Option D: Custom Thresholds

```bash
python monitor.py --ear 0.22 --mar 0.70
```

### ▶️ Option E: Disable Session Logging

```bash
python monitor.py --no-log
```

### All CLI Arguments

```
--source     Video source: 0 (webcam), 1, or video file path  [default: 0]
--ear        EAR drowsiness threshold                         [default: 0.25]
--mar        MAR yawn threshold                               [default: 0.65]
--calibrate  Auto-calibrate EAR from live camera
--no-log     Disable CSV session logging
```

### Keyboard Shortcuts (while running)

| Key | Action |
|-----|--------|
| `q` | Quit monitor |
| `s` | Save screenshot |
| `r` | Reset session counters |
| `c` | Toggle landmark mesh overlay |

---

## 📊 Post-Session Analysis

After running the monitor, analyze the session log:

```bash
# Analyze the most recent session
python analyze_session.py

# Analyze a specific log
python analyze_session.py --log logs/session_20260912_103045.csv

# Analyze all sessions
python analyze_session.py --all
```

Generates a report in `plots/` with:
- 📈 EAR over time with blink events
- 📈 MAR over time with yawn events  
- 📈 Head pose angles (pitch, yaw, roll)
- 🥧 Status distribution pie chart
- 📊 Alert level timeline
- 📋 Summary statistics

---

## 🎮 Demo Without Webcam

No webcam? Generate synthetic data and run the full pipeline:

```bash
# Generate 2-minute simulated session + run analysis
python demo_simulate.py --duration 120 --plot
```

This creates `logs/session_demo.csv` and saves a chart in `plots/`.

---

## 🔬 How It Works

```
Webcam Frame (30 FPS)
        │
        ▼
  ┌─────────────────────────┐
  │  MediaPipe Face Mesh    │  ← Google's pre-trained model
  │  468 3D facial landmarks│     (runs on CPU, no GPU needed)
  └─────────┬───────────────┘
            │
     ┌──────┴──────┐
     │             │
     ▼             ▼
  EAR / MAR    Head Pose (solvePnP)
  (eye/mouth   (pitch, yaw, roll
  geometry)     from 6 landmarks)
     │             │
     └──────┬──────┘
            │
       State Machine
       (ALERT → CAUTION → WARNING → DANGER)
            │
    ┌───────┴──────────┐
    │                  │
   HUD              Audio Alert
  Overlay          (pygame beep)
    │
  CSV Logger
  (per-frame metrics)
```

### Eye Aspect Ratio (EAR) Formula

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)

p1 ─── p2 ─── p3
│               │
p6 ─── p5 ─── p4

Open eye  → EAR ≈ 0.30
Closed eye → EAR ≈ 0.05
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|---------|
| `ModuleNotFoundError: mediapipe` | `pip install mediapipe` |
| `No module named 'cv2'` | `pip install opencv-python` |
| Webcam not found | Try `--source 1` or `--source 2` |
| No audio alerts | `pip install pygame` |
| Low detection accuracy | Use `--calibrate` flag |
| MediaPipe install fails | Use Python 3.10: `py -3.10 -m venv venv` |

---

## 📚 References & Papers

- **MediaPipe Face Mesh** — Kartynnik et al., *Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs*, CVPR 2019
- **EAR for Drowsiness** — Soukupová & Čech, *Real-Time Eye Blink Detection using Facial Landmarks*, CVWW 2016
- **Head Pose via solvePnP** — OpenCV documentation
- **MAR for Yawning** — Abtahi et al., *YAWNet: Detecting Yawning and Pose in the Wild*, 2014

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

*Developed as a Computer Vision course project — AI Driver Safety Monitor using MediaPipe + OpenCV.*
