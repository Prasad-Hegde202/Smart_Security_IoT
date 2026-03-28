from flask import Flask, request, jsonify, send_from_directory
import os
import face_recognition
import pickle
import numpy as np
import requests
import sqlite3
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Config from environment variables ─────────────────────────────────────
BOT_TOKEN   = os.environ.get("BOT_TOKEN", "8693553244:AAE4VtpVd2S7HOtk-wdWnbqcBW1no12S50g")
CHAT_ID     = os.environ.get("CHAT_ID", "6580212381")

# ── Persistent storage paths ───────────────────────────────────────────────
# On Render free tier, use /tmp (survives the session, not reboots)
# For true persistence, mount a Render Disk (paid) at /data
DATA_DIR    = os.environ.get("DATA_DIR", "/tmp/sentinel")
UPLOAD_DIR  = os.path.join(DATA_DIR, "uploads")
DB_PATH     = os.path.join(DATA_DIR, "alerts.db")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Load known face encodings ──────────────────────────────────────────────
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

# ── Database ───────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT NOT NULL,
            status     TEXT NOT NULL,
            image_path TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── Telegram ───────────────────────────────────────────────────────────────
def send_telegram_alert(image_path):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured, skipping alert.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(image_path, "rb") as img:
        requests.post(url, data={
            "chat_id": CHAT_ID,
            "caption": "🚨 Unknown person detected!"
        }, files={"photo": img})

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return "Sentinel Security Backend Running 🚀"

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file     = request.files["image"]
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    image         = face_recognition.load_image_file(filepath)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    results = []
    for encoding in face_encodings:
        name = "Unknown"

        if len(data["encodings"]) > 0:
            matches      = face_recognition.compare_faces(data["encodings"], encoding)
            distances    = face_recognition.face_distance(data["encodings"], encoding)
            best_idx     = int(np.argmin(distances))
            if matches[best_idx]:
                name = data["names"][best_idx]

        if name == "Unknown":
            send_telegram_alert(filepath)
            conn = get_db()
            conn.execute(
                "INSERT INTO alerts (timestamp, status, image_path) VALUES (?, ?, ?)",
                (datetime.now().isoformat(), "Unknown", f"uploads/{filename}")
            )
            conn.commit()
            conn.close()

        results.append(name)

    return jsonify({"faces_detected": len(results), "results": results})

@app.route("/alerts")
def get_alerts():
    conn  = get_db()
    rows  = conn.execute("SELECT * FROM alerts ORDER BY id DESC").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/uploads/<filename>")
def serve_image(filename):
    return send_from_directory(UPLOAD_DIR, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
```

