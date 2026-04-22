"""
OCULUS — Eye Gaze Emotion Engine
Flask Backend · LIVE TOBII VERSION
+ Video Stimulus with AUDIO  (python-vlc)
+ Gaze Cursor Overlay         (OpenCV transparent window on top)
+ Full Dashboard Support
+ Stop Recording + Report Download

Requirements:
    pip install python-vlc opencv-python joblib flask tobii_research
    VLC media player must be installed on the machine:
        https://www.videolan.org/vlc/
"""

import numpy as np
import cv2
import base64
import joblib
import os
import time
import threading
import json
import csv
import io
from datetime import datetime
from collections import deque
from flask import Flask, jsonify, send_from_directory, Response
import tobii_research as tr
import vlc  # pip install python-vlc  (VLC app must also be installed)

app = Flask(__name__)

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
MODEL_PATH = r"C:\Users\HP\Desktop\DSU\real_time_tracking\xgb_saccade_model.pkl"
VIDEO_PATH = r"C:\Users\HP\Desktop\DSU\stimuli\horror.mp4"

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
loaded        = joblib.load(MODEL_PATH)
model         = loaded[0]
label_encoder = loaded[1]
feature_names = loaded[2]
scaler        = loaded[3]

# ─────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────
state = {
    "emotion": "—",
    "confidence": 0.0,
    "probs": {},
    "true_label": "LIVE",
    "stimulus": "VIDEO",

    "frame": 0,
    "total_frames": 0,
    "timestamp": 0.0,

    "avg_x": 0.0,
    "avg_y": 0.0,
    "std_x": 0.0,
    "std_y": 0.0,
    "avg_pupil": 0.0,
    "saccade_speed": 0.0,

    "scanpath_x": [],
    "scanpath_y": [],
    "heatmap_b64": "",

    "running":   True,
    "recording": False,
    "stopped":   False,

    "n_classes": len(label_encoder.classes_),
    "classes":   list(label_encoder.classes_),

    "tobii_status":  "Connecting…",
    "boot_log":      [],
    "session_start": None,
    "session_end":   None,
}

lock = threading.Lock()

# ─────────────────────────────────────────
# REPORT LOG
# ─────────────────────────────────────────
report_log = []

def log_entry(snapshot):
    report_log.append({
        "timestamp":     snapshot.get("timestamp"),
        "frame":         snapshot.get("frame"),
        "emotion":       snapshot.get("emotion"),
        "confidence":    snapshot.get("confidence"),
        "avg_x":         snapshot.get("avg_x"),
        "avg_y":         snapshot.get("avg_y"),
        "std_x":         snapshot.get("std_x"),
        "std_y":         snapshot.get("std_y"),
        "avg_pupil":     snapshot.get("avg_pupil"),
        "saccade_speed": snapshot.get("saccade_speed"),
        **{f"prob_{cls}": snapshot["probs"].get(cls, 0.0)
           for cls in label_encoder.classes_},
    })

def boot(msg, status="ok"):
    with lock:
        state["boot_log"].append({"msg": msg, "status": status})
    print(f"  [boot] {msg}")

# ─────────────────────────────────────────
# BUFFERS
# ─────────────────────────────────────────
WINDOW_SIZE = 30

x_buf     = deque(maxlen=2000)
y_buf     = deque(maxlen=2000)
pupil_buf = deque(maxlen=2000)
t_buf     = deque(maxlen=2000)
frame_count = [0]

# ─────────────────────────────────────────
# TOBII CALLBACK
# ─────────────────────────────────────────
def gaze_data_callback(gaze_data):
    if state.get("stopped"):
        return
    try:
        lgp = gaze_data["left_gaze_point_on_display_area"]
        rgp = gaze_data["right_gaze_point_on_display_area"]

        lx, ly = float(lgp[0]), float(lgp[1])
        rx, ry = float(rgp[0]), float(rgp[1])

        if not (0 <= lx <= 1 and 0 <= ly <= 1 and 0 <= rx <= 1 and 0 <= ry <= 1):
            return

        x = (lx + rx) / 2
        y = (ly + ry) / 2

        lp = gaze_data["left_pupil_diameter"]
        rp = gaze_data["right_pupil_diameter"]

        if np.isnan(lp) or np.isnan(rp):
            return

        pupil = np.mean([lp, rp])
        t = gaze_data["system_time_stamp"] / 1_000_000.0

        x_buf.append(x)
        y_buf.append(y)
        pupil_buf.append(pupil)
        t_buf.append(t)
        frame_count[0] += 1

    except:
        pass

# ─────────────────────────────────────────
# FEATURE EXTRACTION  (unchanged)
# ─────────────────────────────────────────
def extract_features(x, y, pupil, t):
    x = np.array(x); y = np.array(y)
    pupil = np.array(pupil); t = np.array(t)

    if len(x) < 5:
        return None

    avg_x = np.mean(x); avg_y = np.mean(y)
    std_x = np.std(x);  std_y = np.std(y)
    avg_pupil = np.mean(pupil)

    speeds = []
    for i in range(1, len(x)):
        dist = np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)
        dt   = t[i] - t[i-1]
        if dt > 0:
            speeds.append(dist / dt)

    saccade_speed = np.mean(speeds) if speeds else 0
    return avg_x, avg_y, std_x, std_y, avg_pupil, saccade_speed

# ─────────────────────────────────────────
# HEATMAP  (unchanged)
# ─────────────────────────────────────────
def generate_heatmap(x, y):
    width = 800; height = 600
    heatmap = np.zeros((height, width))

    for xi, yi in zip(x, y):
        px = int(xi * width); py = int(yi * height)
        if 0 <= px < width and 0 <= py < height:
            heatmap[py][px] += 1

    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    _, buf = cv2.imencode(".jpg", heatmap)
    return base64.b64encode(buf).decode()

# ─────────────────────────────────────────
# VIDEO STIMULUS WITH AUDIO  (VLC)
# + GAZE CURSOR OVERLAY       (OpenCV)
#
# Architecture:
#   VLC  → plays video + audio in its own window (handles all codecs/audio)
#   OpenCV → separate transparent-background window on top showing only
#            the green gaze cursor dot, updated at ~60 Hz
#
# Why separate windows?
#   VLC renders internally and doesn't expose a per-frame callback that's
#   easy to draw on. The overlay approach is simpler and more reliable.
# ─────────────────────────────────────────

# Window dimensions — match your display or make fullscreen
WIN_W = 800
WIN_H = 600
OVERLAY_WIN = "Gaze Overlay"   # OpenCV window name for cursor

def stimulus_loop():
    """
    Plays the video with audio via VLC, then runs an OpenCV loop
    that draws the gaze cursor in a separate always-on-top window.
    Both windows are positioned/sized to overlap so it looks like
    one single stimulus display with a cursor.
    """
    if not os.path.exists(VIDEO_PATH):
        boot(f"Video not found: {VIDEO_PATH}", "error")
        return

    boot("Initialising VLC media player…", "ok")

    # ── 1. Create VLC instance and media player ──
    vlc_instance = vlc.Instance("--no-xlib")   # --no-xlib avoids X11 issues on some systems
    player       = vlc_instance.media_player_new()
    media        = vlc_instance.media_new(VIDEO_PATH)
    player.set_media(media)

    # Set playback window size
    player.video_set_scale(0)   # 0 = fit to window

    boot("Stimulus video loaded ✓  |  Audio enabled", "ok")

    # ── 2. Start VLC playback (audio + video) ──
    player.play()
    time.sleep(0.5)   # give VLC a moment to open its window

    # Resize VLC window (works on Windows via win32 or just via VLC itself)
    player.video_set_scale(0)

    boot("Video playing with audio ✓", "ok")

    # ── 3. OpenCV overlay window for gaze cursor ──
    # Create a named window that floats on top
    cv2.namedWindow(OVERLAY_WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(OVERLAY_WIN, WIN_W, WIN_H)
    cv2.setWindowProperty(OVERLAY_WIN, cv2.WND_PROP_TOPMOST, 1)

    # On Windows, move it to (0,0) to align with VLC window
    # Adjust these coordinates if your VLC window appears elsewhere
    cv2.moveWindow(OVERLAY_WIN, 0, 0)

    # ── 4. Cursor draw loop ──
    while state["running"] and not state["stopped"]:

        # Check if VLC finished playing
        if player.get_state() == vlc.State.Ended:
            break

        # Transparent black canvas
        overlay = np.zeros((WIN_H, WIN_W, 4), dtype=np.uint8)

        # Draw gaze cursor if we have data
        if len(x_buf) > 0:
            gx = float(x_buf[-1])
            gy = float(y_buf[-1])

            # Clamp to window bounds
            gx = max(0.0, min(1.0, gx))
            gy = max(0.0, min(1.0, gy))

            cx = int(gx * WIN_W)
            cy = int(gy * WIN_H)

            # Outer glow ring (semi-transparent green)
            cv2.circle(overlay, (cx, cy), 22, (0, 255, 80, 60),  2, cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), 16, (0, 255, 80, 100), 2, cv2.LINE_AA)
            # Solid inner dot
            cv2.circle(overlay, (cx, cy), 8,  (0, 255, 80, 220), -1, cv2.LINE_AA)
            # White centre pinpoint
            cv2.circle(overlay, (cx, cy), 2,  (255, 255, 255, 255), -1, cv2.LINE_AA)

            # Small crosshair lines
            cv2.line(overlay, (cx-28, cy), (cx-12, cy), (0, 255, 80, 140), 1, cv2.LINE_AA)
            cv2.line(overlay, (cx+12, cy), (cx+28, cy), (0, 255, 80, 140), 1, cv2.LINE_AA)
            cv2.line(overlay, (cx, cy-28), (cx, cy-12), (0, 255, 80, 140), 1, cv2.LINE_AA)
            cv2.line(overlay, (cx, cy+12), (cx, cy+28), (0, 255, 80, 140), 1, cv2.LINE_AA)

        # Show overlay (BGR only — OpenCV doesn't support true alpha blend in imshow,
        # but the black background will appear as a dark tint; for a truly transparent
        # overlay you'd use a Win32 layered window — see NOTE below)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_BGRA2BGR)
        cv2.imshow(OVERLAY_WIN, overlay_bgr)

        key = cv2.waitKey(16) & 0xFF   # ~60 fps
        if key == 27:   # ESC to exit
            break

    # ── 5. Cleanup ──
    player.stop()
    cv2.destroyAllWindows()
    boot("Stimulus playback ended", "ok")


# ─────────────────────────────────────────
# PROCESSING LOOP  (unchanged)
# ─────────────────────────────────────────
def processing_loop(eyetracker):
    import pandas as pd

    while state["running"] and not state["stopped"]:
        if len(x_buf) < WINDOW_SIZE:
            time.sleep(0.05)
            continue

        x_list = list(x_buf)[-WINDOW_SIZE:]
        y_list = list(y_buf)[-WINDOW_SIZE:]
        p_list = list(pupil_buf)[-WINDOW_SIZE:]
        t_list = list(t_buf)[-WINDOW_SIZE:]

        feats = extract_features(x_list, y_list, p_list, t_list)

        if feats:
            avg_x, avg_y, std_x, std_y, avg_pupil, saccade_speed = feats

            df = pd.DataFrame(
                [[avg_x, avg_y, std_x, std_y, avg_pupil, saccade_speed]],
                columns=feature_names
            )

            sc      = scaler.transform(df.values)
            pred    = model.predict(sc)[0]
            emotion = label_encoder.inverse_transform([pred])[0]
            probs   = model.predict_proba(sc)[0]

            prob_dict = {cls: float(p) for cls, p in zip(label_encoder.classes_, probs)}
            conf      = float(np.max(probs)) * 100

            heatmap_b64 = generate_heatmap(x_list, y_list)

            snapshot = {
                "emotion":       emotion,
                "confidence":    round(conf, 1),
                "probs":         prob_dict,
                "frame":         frame_count[0],
                "total_frames":  frame_count[0],
                "timestamp":     round(t_list[-1], 3),
                "avg_x":         round(avg_x, 5),
                "avg_y":         round(avg_y, 5),
                "std_x":         round(std_x, 5),
                "std_y":         round(std_y, 5),
                "avg_pupil":     round(avg_pupil, 3),
                "saccade_speed": round(saccade_speed, 4),
                "scanpath_x":    x_list[-40:],
                "scanpath_y":    y_list[-40:],
                "heatmap_b64":   heatmap_b64,
            }

            with lock:
                state.update(snapshot)

            log_entry(snapshot)

        time.sleep(0.2)

    eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
    print("[processing] Stopped.")

# ─────────────────────────────────────────
# CONNECT TO TOBII  (unchanged)
# ─────────────────────────────────────────
def connect_tobii():
    boot("Initialising OCULUS system…")
    time.sleep(0.4)
    boot("Loading XGBoost classifier…")
    time.sleep(0.3)
    boot(f"Feature space: {list(feature_names)}")
    time.sleep(0.3)
    boot("Scanning for Tobii Pro device…")

    trackers = tr.find_all_eyetrackers()

    if not trackers:
        boot("No Tobii tracker detected!", "error")
        raise RuntimeError("No Tobii tracker detected")

    tracker = trackers[0]
    boot(f"Tobii connected: {tracker.model}  |  S/N {tracker.serial_number}", "ok")
    time.sleep(0.2)
    boot(f"Sampling frequency: {tracker.get_gaze_output_frequency()} Hz", "ok")
    time.sleep(0.2)
    boot("Gaze data stream subscribed ✓", "ok")
    time.sleep(0.2)
    boot("All systems nominal — launching OCULUS dashboard…", "ready")

    with lock:
        state["tobii_status"]  = f"Connected · {tracker.model}"
        state["recording"]     = True
        state["session_start"] = datetime.now().isoformat()

    return tracker

# ─────────────────────────────────────────
# API ROUTES  (unchanged)
# ─────────────────────────────────────────
@app.route("/api/state")
def get_state():
    with lock:
        data = dict(state)
    resp = jsonify(data)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

@app.route("/api/stop", methods=["POST"])
def stop_recording():
    with lock:
        state["stopped"]     = True
        state["recording"]   = False
        state["session_end"] = datetime.now().isoformat()
    resp = jsonify({"ok": True, "entries": len(report_log)})
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

@app.route("/api/report/csv")
def download_csv():
    if not report_log:
        return "No data recorded yet.", 404

    si = io.StringIO()
    w  = csv.DictWriter(si, fieldnames=report_log[0].keys())
    w.writeheader()
    w.writerows(report_log)

    resp = Response(si.getvalue(), mimetype="text/csv")
    resp.headers["Content-Disposition"] = \
        f'attachment; filename="oculus_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

@app.route("/api/report/json")
def download_json():
    if not report_log:
        return "No data recorded yet.", 404

    with lock:
        payload = {
            "session_start": state.get("session_start"),
            "session_end":   state.get("session_end"),
            "total_entries": len(report_log),
            "classes":       list(label_encoder.classes_),
            "records":       report_log,
        }

    resp = Response(json.dumps(payload, indent=2), mimetype="application/json")
    resp.headers["Content-Disposition"] = \
        f'attachment; filename="oculus_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"'
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n  ╔══════════════════════════════════════╗")
    print("  ║   OCULUS  ·  Live Tobii Edition      ║")
    print("  ╚══════════════════════════════════════╝\n")

    eyetracker = connect_tobii()

    eyetracker.subscribe_to(
        tr.EYETRACKER_GAZE_DATA,
        gaze_data_callback,
        as_dictionary=True
    )

    proc_thread = threading.Thread(
        target=processing_loop, args=(eyetracker,), daemon=True
    )
    proc_thread.start()

    # stimulus_loop() must run on the MAIN thread on Windows
    # because OpenCV's imshow requires the main thread for GUI on Windows.
    # So we run Flask in a background thread instead.
    flask_thread = threading.Thread(
        target=lambda: app.run(port=5050, debug=False, use_reloader=False),
        daemon=True
    )
    flask_thread.start()

    print("  Server → http://localhost:5050")
    print("  Open dashboard.html in your browser\n")

    # Run stimulus (VLC + OpenCV cursor) on main thread
    stimulus_loop()

    # After video ends, keep Flask alive
    print("\n  Video ended. Flask server still running — press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Shutting down.")
        with lock:
            state["running"] = False