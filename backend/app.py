from flask import Flask, request, jsonify, send_from_directory
import os
import requests
import sqlite3
from datetime import datetime
from flask_cors import CORS
from deepface import DeepFace   # no dlib, no cmake, no compilation

app = Flask(__name__)
CORS(app)

# ── Environment detection ──────────────────────────────────────────────────
IS_CLOUD = bool(os.environ.get("RENDER", False))

# ── Paths ──────────────────────────────────────────────────────────────────
if IS_CLOUD:
    BASE_DIR = os.environ.get("DATA_DIR", "/tmp/sentinel")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DB_PATH    = os.path.join(BASE_DIR, "alerts.db")

# Known faces: just drop photos here — no encoding or training needed
# e.g.  known_faces/john.jpg   known_faces/jane.jpg
# Name shown in results = filename without extension
KNOWN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_faces")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(KNOWN_DIR,  exist_ok=True)

print(f"[sentinel] Mode       : {'CLOUD' if IS_CLOUD else 'LOCAL'}")
print(f"[sentinel] Uploads    : {UPLOAD_DIR}")
print(f"[sentinel] Database   : {DB_PATH}")
print(f"[sentinel] Known faces: {KNOWN_DIR}")

# ── Telegram ───────────────────────────────────────────────────────────────
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID   = os.environ.get("CHAT_ID",   "")

def send_telegram_alert(image_path):
    if not BOT_TOKEN or not CHAT_ID:
        print("[sentinel] Telegram not configured — skipping")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        with open(image_path, "rb") as img:
            resp = requests.post(
                url,
                data={"chat_id": CHAT_ID, "caption": "🚨 Unknown person detected!"},
                files={"photo": img},
                timeout=10
            )
        print(f"[sentinel] Telegram: {resp.status_code}")
    except Exception as e:
        print(f"[sentinel] Telegram error: {e}")

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

# ── Face check via DeepFace ────────────────────────────────────────────────
def is_known_face(image_path):
    """
    Returns (True, name)  — face matches someone in known_faces/
    Returns (False, "Unknown") — no match or no known faces registered

    DeepFace.find() compares the uploaded photo against every image
    in known_faces/ automatically. Zero training required.
    """
    known_photos = [
        f for f in os.listdir(KNOWN_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not known_photos:
        print("[sentinel] known_faces/ is empty — all faces marked Unknown")
        return False, "Unknown"

    try:
        results = DeepFace.find(
            img_path          = image_path,
            db_path           = KNOWN_DIR,
            model_name        = "Facenet",   # lightweight & accurate
            detector_backend  = "opencv",    # fastest, no extra deps
            enforce_detection = False,       # don't crash on unclear faces
            silent            = True
        )

        # results = list of DataFrames, one per detected face
        if results and not results[0].empty:
            matched_path = results[0].iloc[0]["identity"]
            name = os.path.splitext(os.path.basename(matched_path))[0]
            print(f"[sentinel] Matched known face: {name}")
            return True, name

    except Exception as e:
        print(f"[sentinel] DeepFace error: {e}")

    return False, "Unknown"

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({
        "status":      "running",
        "environment": "cloud" if IS_CLOUD else "local"
    })

@app.route("/health")
def health():
    known_count = len([
        f for f in os.listdir(KNOWN_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return jsonify({
        "status":      "ok",
        "environment": "cloud" if IS_CLOUD else "local",
        "known_faces": known_count,
        "upload_dir":  UPLOAD_DIR,
        "db_path":     DB_PATH,
    })

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file     = request.files["image"]
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    known, name = is_known_face(filepath)

    if not known:
        send_telegram_alert(filepath)
        conn = get_db()
        conn.execute(
            "INSERT INTO alerts (timestamp, status, image_path) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), "Unknown", f"uploads/{filename}")
        )
        conn.commit()
        conn.close()
        print("[sentinel] Unknown face — alert saved + Telegram sent")
    else:
        print(f"[sentinel] Known face ({name}) — no alert")

    return jsonify({
        "face_detected": True,
        "name":          name,
        "alert_sent":    not known
    })

@app.route("/alerts")
def get_alerts():
    conn = get_db()
    rows = conn.execute("SELECT * FROM alerts ORDER BY id DESC").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/uploads/<filename>")
def serve_image(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# ── Register a known face via API ──────────────────────────────────────────
# Useful when running on cloud where you can't drop files manually
@app.route("/known-faces/add", methods=["POST"])
def add_known_face():
    """
    POST form-data with:
      image = <photo file>
      name  = "john_doe"
    Saves to known_faces/john_doe.jpg
    """
    if "image" not in request.files or "name" not in request.form:
        return jsonify({"error": "Provide 'image' file and 'name' text field"}), 400

    name     = request.form["name"].strip().replace(" ", "_")
    file     = request.files["image"]
    savepath = os.path.join(KNOWN_DIR, f"{name}.jpg")
    file.save(savepath)
    return jsonify({"message": f"Added known face: {name}", "path": savepath})

@app.route("/known-faces")
def list_known_faces():
    names = [
        os.path.splitext(f)[0]
        for f in os.listdir(KNOWN_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return jsonify({"known_faces": names, "count": len(names)})

# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = not IS_CLOUD
    print(f"[sentinel] Starting on port {port}  debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)