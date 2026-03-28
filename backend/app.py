from flask import Flask, request, jsonify, send_from_directory
import os
import requests
import sqlite3
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Environment ────────────────────────────────────────────────────────────
IS_CLOUD = bool(os.environ.get("RENDER", False))

if IS_CLOUD:
    BASE_DIR = os.environ.get("DATA_DIR", "/tmp/sentinel")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DB_PATH    = os.path.join(BASE_DIR, "alerts.db")
KNOWN_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_faces")

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
                data={"chat_id": CHAT_ID, "caption": f"🚨 {unknown_count} unknown person(s) detected!"},
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

# ── DeepFace — lazy import + model pre-warm ────────────────────────────────
# Import is deferred so the app starts fast.
# We then pre-download/cache the model ONCE at startup
# so the first real request doesn't time out.
print("[sentinel] Pre-loading DeepFace model...")
try:
    # Use the lightest model: VGG-Face (~500MB less than Facenet on RAM)
    # opencv detector — no extra deps, fastest
    from deepface import DeepFace
    import numpy as np
    import cv2

    # Warm up: build a tiny blank image and run a dummy verify
    # This forces model weights to download & cache now, not on first request
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_path = "/tmp/_sentinel_warmup.jpg"
    cv2.imwrite(dummy_path, dummy)

    try:
        DeepFace.represent(
            img_path          = dummy_path,
            model_name        = "VGG-Face",
            detector_backend  = "opencv",
            enforce_detection = False
        )
    except Exception:
        pass  # expected to fail on blank image, but model is now cached

    os.remove(dummy_path)
    print("[sentinel] DeepFace model ready ✓")
    DEEPFACE_READY = True

except Exception as e:
    print(f"[sentinel] DeepFace failed to load: {e}")
    DEEPFACE_READY = False

# ── Face recognition ───────────────────────────────────────────────────────
def recognize_faces(image_path):
    """
    Returns list of names for every face in the image.
    e.g. ["prasad", "Unknown"]
    """
    if not DEEPFACE_READY:
        print("[sentinel] DeepFace not ready — returning Unknown")
        return ["Unknown"]

    known_photos = [
        f for f in os.listdir(KNOWN_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    results = []

    try:
        # Step 1: detect all faces
        faces = DeepFace.extract_faces(
            img_path          = image_path,
            detector_backend  = "opencv",
            enforce_detection = False
        )

        if not faces:
            print("[sentinel] No faces detected")
            return []

        print(f"[sentinel] {len(faces)} face(s) detected")

        # Step 2: identify each face
        for i, face_obj in enumerate(faces):
            name = "Unknown"

            if known_photos:
                try:
                    # Save cropped face to temp file for matching
                    face_pixels = (face_obj["face"] * 255).astype(np.uint8)
                    temp_path   = os.path.join(UPLOAD_DIR, f"_tmp_face_{i}.jpg")
                    cv2.imwrite(temp_path, cv2.cvtColor(face_pixels, cv2.COLOR_RGB2BGR))

                    match_results = DeepFace.find(
                        img_path          = temp_path,
                        db_path           = KNOWN_DIR,
                        model_name        = "VGG-Face",   # lighter than Facenet
                        detector_backend  = "opencv",
                        enforce_detection = False,
                        silent            = True
                    )

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                    if match_results and not match_results[0].empty:
                        matched_path = match_results[0].iloc[0]["identity"]
                        name = os.path.splitext(os.path.basename(matched_path))[0]
                        print(f"[sentinel] Face {i+1} → {name}")
                    else:
                        print(f"[sentinel] Face {i+1} → Unknown")

                except Exception as e:
                    print(f"[sentinel] Face {i+1} match error: {e}")
            else:
                print(f"[sentinel] Face {i+1} → Unknown (no known faces)")

            results.append(name)

    except Exception as e:
        print(f"[sentinel] extract_faces error: {e}")

    return results

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({
        "status":        "running",
        "environment":   "cloud" if IS_CLOUD else "local",
        "deepface_ready": DEEPFACE_READY
    })

@app.route("/health")
def health():
    known_count = len([
        f for f in os.listdir(KNOWN_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return jsonify({
        "status":        "ok",
        "environment":   "cloud" if IS_CLOUD else "local",
        "deepface_ready": DEEPFACE_READY,
        "known_faces":   known_count,
    })

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    if not DEEPFACE_READY:
        return jsonify({"error": "Face recognition model not loaded yet, try again in 30 seconds"}), 503

    file     = request.files["image"]
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    results       = recognize_faces(filepath)
    unknown_count = results.count("Unknown")
    has_unknown   = unknown_count > 0

    if has_unknown:
        send_telegram_alert(filepath, unknown_count)
        conn = get_db()
        conn.execute(
            "INSERT INTO alerts (timestamp, status, image_path) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), "Unknown", f"uploads/{filename}")
        )
        conn.commit()
        conn.close()
        print(f"[sentinel] Alert saved — {unknown_count} unknown face(s)")
    else:
        print(f"[sentinel] All known: {results}")

    return jsonify({
        "faces_detected": len(results),
        "results":        results
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

@app.route("/known-faces/add", methods=["POST"])
def add_known_face():
    if "image" not in request.files or "name" not in request.form:
        return jsonify({"error": "Provide 'image' file and 'name' text field"}), 400
    name     = request.form["name"].strip().replace(" ", "_").lower()
    file     = request.files["image"]
    savepath = os.path.join(KNOWN_DIR, f"{name}.jpg")
    file.save(savepath)
    print(f"[sentinel] Registered: {name}")
    return jsonify({"message": f"Known face '{name}' registered", "path": savepath})

@app.route("/known-faces")
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
        return jsonify({"message": f"Deleted: {name}"})
    return jsonify({"error": "Not found"}), 404

# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = not IS_CLOUD
    print(f"[sentinel] Starting on port {port}  debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)