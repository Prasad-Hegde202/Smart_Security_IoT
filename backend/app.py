from flask import Flask, request, jsonify, send_from_directory
import os
import requests
import sqlite3
from datetime import datetime
from flask_cors import CORS
from deepface import DeepFace

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

# known_faces/ is always relative to app.py so it works locally AND
# on Render as long as you commit the photos to your git repo
# OR upload them via POST /known-faces/add after deploy
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

def send_telegram_alert(image_path, unknown_count):
    if not BOT_TOKEN or not CHAT_ID:
        print("[sentinel] Telegram not configured — skipping")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        with open(image_path, "rb") as img:
            resp = requests.post(
                url,
                data={
                    "chat_id": CHAT_ID,
                    "caption": f"🚨 {unknown_count} unknown person(s) detected!"
                },
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

# ── Face recognition ───────────────────────────────────────────────────────
def recognize_faces(image_path):
    """
    Returns a list of names for every face detected in the image.
    Matches the old face_recognition behavior:
      ["prasad", "Unknown", "Unknown"]
    
    Strategy:
      1. Use DeepFace.extract_faces() to find HOW MANY faces are in the image
      2. For each face, crop it and run DeepFace.find() against known_faces/
      3. Return a name per face
    """
    known_photos = [
        f for f in os.listdir(KNOWN_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    results = []

    try:
        # Step 1: detect all faces in the image
        faces = DeepFace.extract_faces(
            img_path          = image_path,
            detector_backend  = "opencv",
            enforce_detection = False
        )

        if not faces:
            print("[sentinel] No faces detected in image")
            return []

        print(f"[sentinel] Detected {len(faces)} face(s)")

        # Step 2: identify each face
        for i, face_obj in enumerate(faces):
            name = "Unknown"

            # Only try to match if we have known faces registered
            if known_photos:
                try:
                    # Save the cropped face temporarily for matching
                    import cv2
                    import numpy as np

                    face_pixels = (face_obj["face"] * 255).astype(np.uint8)
                    temp_path   = os.path.join(UPLOAD_DIR, f"_temp_face_{i}.jpg")
                    cv2.imwrite(temp_path, cv2.cvtColor(face_pixels, cv2.COLOR_RGB2BGR))

                    match_results = DeepFace.find(
                        img_path          = temp_path,
                        db_path           = KNOWN_DIR,
                        model_name        = "Facenet",
                        detector_backend  = "opencv",
                        enforce_detection = False,
                        silent            = True
                    )

                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                    if match_results and not match_results[0].empty:
                        matched_path = match_results[0].iloc[0]["identity"]
                        name = os.path.splitext(os.path.basename(matched_path))[0]
                        print(f"[sentinel] Face {i+1}: matched → {name}")
                    else:
                        print(f"[sentinel] Face {i+1}: Unknown")

                except Exception as e:
                    print(f"[sentinel] Face {i+1} match error: {e}")
                    name = "Unknown"
            else:
                print(f"[sentinel] Face {i+1}: Unknown (no known faces registered)")

            results.append(name)

    except Exception as e:
        print(f"[sentinel] DeepFace extract error: {e}")

    return results

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

    # Get name for every face in the image — same as old behavior
    results       = recognize_faces(filepath)
    unknown_count = results.count("Unknown")
    has_unknown   = unknown_count > 0

    if has_unknown:
        # Send one Telegram alert per upload (not per face)
        send_telegram_alert(filepath, unknown_count)

        conn = get_db()
        conn.execute(
            "INSERT INTO alerts (timestamp, status, image_path) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), "Unknown", f"uploads/{filename}")
        )
        conn.commit()
        conn.close()
        print(f"[sentinel] {unknown_count} unknown face(s) — alert saved")
    else:
        print(f"[sentinel] All faces known: {results}")

    # ── Response matches old face_recognition format exactly ──────────────
    return jsonify({
        "faces_detected": len(results),
        "results":        results           # e.g. ["prasad", "Unknown"]
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

# ── Known face management ──────────────────────────────────────────────────
@app.route("/known-faces/add", methods=["POST"])
def add_known_face():
    """
    Register a known face.
    POST form-data:
      name  = "prasad"         (text field)
      image = <photo file>     (file field)

    On cloud: use this endpoint after every redeploy since /tmp is wiped.
    Locally:  you can also just drop photos into known_faces/ folder directly.
    """
    if "image" not in request.files or "name" not in request.form:
        return jsonify({"error": "Provide 'image' file and 'name' text field"}), 400

    name     = request.form["name"].strip().replace(" ", "_").lower()
    file     = request.files["image"]
    savepath = os.path.join(KNOWN_DIR, f"{name}.jpg")
    file.save(savepath)

    print(f"[sentinel] Registered known face: {name} → {savepath}")
    return jsonify({
        "message": f"Known face '{name}' registered successfully",
        "path":    savepath
    })

@app.route("/known-faces", methods=["GET"])
def list_known_faces():
    names = [
        os.path.splitext(f)[0]
        for f in os.listdir(KNOWN_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return jsonify({"known_faces": names, "count": len(names)})

@app.route("/known-faces/<name>", methods=["DELETE"])
def delete_known_face(name):
    path = os.path.join(KNOWN_DIR, f"{name}.jpg")
    if os.path.exists(path):
        os.remove(path)
        return jsonify({"message": f"Deleted known face: {name}"})
    return jsonify({"error": "Not found"}), 404

# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = not IS_CLOUD
    print(f"[sentinel] Starting on port {port}  debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)