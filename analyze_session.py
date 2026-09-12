"""
analyze_session.py
-------------------
Post-session analysis: reads CSV log and generates a detailed report with
charts — EAR over time, event timeline, drowsiness heatmap, summary stats.

Usage:
    python analyze_session.py                          # latest session log
    python analyze_session.py --log logs/session_X.csv
    python analyze_session.py --all                    # analyze all sessions
"""

import argparse
import os
import glob
import sys

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds (must match monitor.py)
# ─────────────────────────────────────────────────────────────────────────────
EAR_THRESH = 0.25
MAR_THRESH = 0.65

STATUS_COLORS = {
    "ALERT":   "#27ae60",
    "CAUTION": "#f39c12",
    "WARNING": "#e67e22",
    "DANGER":  "#c0392b",
    "NO FACE": "#7f8c8d",
    "BLINK":   "#3498db",
    "YAWN":    "#9b59b6",
}


def load_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_latest_log(log_dir: str = "logs") -> str:
    logs = sorted(glob.glob(os.path.join(log_dir, "session_*.csv")))
    if not logs:
        print(f"[ERROR] No session logs found in '{log_dir}'.")
        sys.exit(1)
    return logs[-1]


def analyze(log_path: str):
    print(f"[INFO] Analyzing: {log_path}")
    df = load_log(log_path)

    if df.empty:
        print("[WARNING] Log is empty — nothing to analyze.")
        return

    session_name = os.path.splitext(os.path.basename(log_path))[0]
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{session_name}_report.png")

    # ── Stats ──────────────────────────────────────────────────────────────
    duration   = df["elapsed_sec"].max()
    blinks     = (df["event"] == "blink_detected").sum()
    yawns      = (df["event"] == "yawn_detected").sum()
    drowsy_pct = (df["status"].isin(["CAUTION","WARNING","DANGER"]).sum()
                  / len(df) * 100)
    danger_pct = (df["status"] == "DANGER").sum() / len(df) * 100
    avg_ear    = df["ear_avg"].mean()
    avg_mar    = df["mar"].mean()
    blink_rate = blinks / max(duration / 60, 1e-6)

    print(f"\n{'='*55}")
    print(f"  SESSION ANALYSIS REPORT")
    print(f"{'='*55}")
    print(f"  Duration       : {int(duration)//60:02d}:{int(duration)%60:02d}")
    print(f"  Blink count    : {blinks}  ({blink_rate:.1f}/min)")
    print(f"  Yawn count     : {yawns}")
    print(f"  Avg EAR        : {avg_ear:.4f}")
    print(f"  Avg MAR        : {avg_mar:.4f}")
    print(f"  Drowsy frames  : {drowsy_pct:.1f}%")
    print(f"  Danger frames  : {danger_pct:.1f}%")
    print(f"{'='*55}\n")

    # ── Figure ─────────────────────────────────────────────────────────────
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.facecolor": "#1a1a2e",
        "figure.facecolor": "#0f0f23",
        "text.color": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#b0b0b0",
        "ytick.color": "#b0b0b0",
        "axes.edgecolor": "#333355",
        "grid.color": "#252540",
    })

    fig = plt.figure(figsize=(18, 12))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    t = df["elapsed_sec"].values

    # ── 1. EAR over time ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.fill_between(t, df["ear_avg"], alpha=0.25, color="#00b4d8")
    ax1.plot(t, df["ear_avg"],  color="#00b4d8", linewidth=1.2, label="Avg EAR")
    ax1.plot(t, df["ear_left"], color="#48cae4", linewidth=0.6,
             alpha=0.5, linestyle="--", label="L-EAR")
    ax1.plot(t, df["ear_right"],color="#90e0ef", linewidth=0.6,
             alpha=0.5, linestyle="--", label="R-EAR")
    ax1.axhline(EAR_THRESH, color="#ff6b6b", linewidth=1.5,
                linestyle="--", label=f"Threshold ({EAR_THRESH})")
    ax1.set_title("Eye Aspect Ratio (EAR) Over Time", color="#e0e0e0", fontsize=13)
    ax1.set_xlabel("Elapsed Time (s)")
    ax1.set_ylabel("EAR")
    ax1.legend(fontsize=8, loc="upper right")

    # ── 2. MAR over time ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.fill_between(t, df["mar"], alpha=0.3, color="#c77dff")
    ax2.plot(t, df["mar"], color="#c77dff", linewidth=1.2, label="MAR")
    ax2.axhline(MAR_THRESH, color="#ff9f1c", linewidth=1.5,
                linestyle="--", label=f"Yawn Threshold ({MAR_THRESH})")
    ax2.set_title("Mouth Aspect Ratio (MAR) — Yawn Detection", color="#e0e0e0", fontsize=13)
    ax2.set_xlabel("Elapsed Time (s)")
    ax2.set_ylabel("MAR")
    ax2.legend(fontsize=8)

    # ── 3. Head pose (pitch & yaw) ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.plot(t, df["pitch"], color="#f4a261", linewidth=1.2, label="Pitch (up/down)")
    ax3.plot(t, df["yaw"],   color="#e76f51", linewidth=1.2, label="Yaw (left/right)")
    ax3.plot(t, df["roll"],  color="#8ecae6", linewidth=0.7,
             alpha=0.6, linestyle=":", label="Roll")
    ax3.axhline(20,  color="#f4a261", linewidth=0.8, linestyle="--", alpha=0.5)
    ax3.axhline(-20, color="#f4a261", linewidth=0.8, linestyle="--", alpha=0.5)
    ax3.axhline(35,  color="#e76f51", linewidth=0.8, linestyle="--", alpha=0.5)
    ax3.axhline(-35, color="#e76f51", linewidth=0.8, linestyle="--", alpha=0.5)
    ax3.set_title("Head Pose Angles (degrees)", color="#e0e0e0", fontsize=13)
    ax3.set_xlabel("Elapsed Time (s)")
    ax3.set_ylabel("Degrees")
    ax3.legend(fontsize=8)

    # ── 4. Status distribution (pie) ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 2])
    status_counts = df["status"].value_counts()
    colors_pie = [STATUS_COLORS.get(s, "#888") for s in status_counts.index]
    wedges, texts, autotexts = ax4.pie(
        status_counts.values,
        labels=status_counts.index,
        colors=colors_pie,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"color": "#e0e0e0", "fontsize": 8},
    )
    for at in autotexts:
        at.set_fontsize(7)
    ax4.set_title("Status Distribution", color="#e0e0e0", fontsize=11)

    # ── 5. Alert level timeline ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    alert_colors = ["#27ae60", "#f39c12", "#e67e22", "#c0392b"]
    for lvl in range(4):
        mask = df["alert_level"] == lvl
        ax5.scatter(df.loc[mask, "elapsed_sec"],
                    df.loc[mask, "alert_level"],
                    c=alert_colors[lvl], s=4, alpha=0.7)
    ax5.set_title("Alert Level Timeline", color="#e0e0e0", fontsize=11)
    ax5.set_xlabel("Elapsed (s)")
    ax5.set_ylabel("Alert Level")
    ax5.set_yticks([0, 1, 2, 3])
    ax5.set_yticklabels(["ALERT", "CAUTION", "WARNING", "DANGER"], fontsize=7)

    # ── 6. Summary stats box ───────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis("off")
    stats_text = (
        f"SESSION SUMMARY\n"
        f"{'─'*28}\n"
        f"Duration      : {int(duration)//60:02d}:{int(duration)%60:02d}\n"
        f"Blinks        : {blinks} ({blink_rate:.1f}/min)\n"
        f"Yawns         : {yawns}\n"
        f"Mean EAR      : {avg_ear:.4f}\n"
        f"Mean MAR      : {avg_mar:.4f}\n"
        f"Drowsy Time   : {drowsy_pct:.1f}%\n"
        f"Danger Time   : {danger_pct:.1f}%\n"
        f"{'─'*28}\n"
        + ("⚠ HIGH DROWSINESS DETECTED" if drowsy_pct > 20 else "✓ Session Normal")
    )
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment="top",
             fontfamily="monospace", color="#e0e0e0",
             bbox=dict(boxstyle="round", facecolor="#1a1a2e",
                       edgecolor="#444466", alpha=0.9))

    fig.suptitle(f"Driver Safety Analysis — {session_name}",
                 fontsize=15, color="#ffffff", y=0.98, fontweight="bold")

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Report saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Analyze driver safety session log")
    ap.add_argument("--log", default=None,
                    help="Path to session CSV (defaults to latest in logs/)")
    ap.add_argument("--all", action="store_true",
                    help="Analyze all session logs in logs/")
    args = ap.parse_args()

    if args.all:
        logs = sorted(glob.glob("logs/session_*.csv"))
        if not logs:
            print("[ERROR] No session logs found in logs/")
            sys.exit(1)
        for log in logs:
            analyze(log)
    else:
        log_path = args.log if args.log else get_latest_log()
        analyze(log_path)
