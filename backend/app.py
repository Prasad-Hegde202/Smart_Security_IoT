from flask import Flask, request, jsonify, send_from_directory
import os
import requests
import sqlite3
import numpy as np
import cv2
import pickle
from datetime import datetime
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# ── Environment ────────────────────────────────────────────────────────────
IS_CLOUD = bool(os.environ.get("RENDER", False))

if IS_CLOUD:
    BASE_DIR = os.environ.get("DATA_DIR", "/tmp/sentinel")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR   = os.path.join(BASE_DIR, "uploads")
DB_PATH      = os.path.join(BASE_DIR, "alerts.db")
KNOWN_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_faces")
ENCODINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encodings.pkl")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(KNOWN_DIR,  exist_ok=True)

print(f"[sentinel] Mode    : {'CLOUD' if IS_CLOUD else 'LOCAL'}")
print(f"[sentinel] Uploads : {UPLOAD_DIR}")
print(f"[sentinel] Database: {DB_PATH}")

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

# ── Load InsightFace ───────────────────────────────────────────────────────
# InsightFace uses ONNX runtime — no TensorFlow, no PyTorch
# RAM usage: ~150MB vs TensorFlow's 500MB+
print("[sentinel] Loading InsightFace model...")
try:
    import insightface
    from insightface.app import FaceAnalysis

    face_app = FaceAnalysis(
        name       = "buffalo_sc",   # smallest model: detection + recognition
        providers  = ["CPUExecutionProvider"]  # CPU only (no GPU on Render)
    )
    face_app.prepare(ctx_id=0, det_size=(320, 320))  # 320 instead of 640 = less RAM
    print("[sentinel] InsightFace ready ✓")
    FACE_APP_READY = True

except Exception as e:
    print(f"[sentinel] InsightFace failed to load: {e}")
    face_app = None
    FACE_APP_READY = False

# ── Known face encodings ───────────────────────────────────────────────────
# Stored as { name: embedding_vector }
# Built from photos in known_faces/ folder
# Saved to encodings.pkl so it persists across requests

def load_encodings():
    """Load saved encodings from disk."""
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}   # { "prasad": np.array([...]) }

def save_encodings(encodings):
    """Save encodings to disk."""
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(encodings, f)

def get_face_embedding(image_path):
    """Get face embedding vector from an image. Returns None if no face found."""
    if not FACE_APP_READY:
        return None
    try:
        img   = cv2.imread(image_path)
        faces = face_app.get(img)
        if not faces:
            return None
        # Return embedding of the largest (most prominent) face
        largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        return largest.embedding
    except Exception as e:
        print(f"[sentinel] Embedding error: {e}")
        return None

def rebuild_encodings_from_folder():
    """
    Scan known_faces/ folder and build embeddings for all photos.
    Call this after adding new known faces.
    """
    if not FACE_APP_READY:
        return {}

    encodings = {}
    photos = [f for f in os.listdir(KNOWN_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for photo in photos:
        name  = os.path.splitext(photo)[0]
        path  = os.path.join(KNOWN_DIR, photo)
        emb   = get_face_embedding(path)
        if emb is not None:
            encodings[name] = emb
            print(f"[sentinel] Encoded known face: {name}")
        else:
            print(f"[sentinel] No face found in {photo} — skipping")

    save_encodings(encodings)
    return encodings

# Load existing encodings on startup
known_encodings = load_encodings()
print(f"[sentinel] Loaded {len(known_encodings)} known face encoding(s)")

# If encodings.pkl is empty but known_faces/ has photos, rebuild
if not known_encodings:
    known_photos = [f for f in os.listdir(KNOWN_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))]
    if known_photos:
        print("[sentinel] Rebuilding encodings from known_faces/ folder...")
        known_encodings = rebuild_encodings_from_folder()

# ── Face recognition ───────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.4   # tune: higher = stricter matching

def recognize_faces(image_path):
    """
    Returns list of names for every face in the image.
    e.g. ["prasad", "Unknown"]
    Matches old face_recognition response format exactly.
    """
    if not FACE_APP_READY:
        return ["Unknown"]

    try:
        img   = cv2.imread(image_path)
        faces = face_app.get(img)

        if not faces:
            print("[sentinel] No faces detected")
            return []

        print(f"[sentinel] {len(faces)} face(s) detected")
        results = []

        for i, face in enumerate(faces):
            name = "Unknown"

            if known_encodings:
                query_emb = face.embedding.reshape(1, -1)

                # Compare against all known faces
                best_name  = None
                best_score = -1

                for known_name, known_emb in known_encodings.items():
                    score = cosine_similarity(query_emb, known_emb.reshape(1, -1))[0][0]
                    if score > best_score:
                        best_score = score
                        best_name  = known_name

                if best_score >= SIMILARITY_THRESHOLD:
                    name = best_name
                    print(f"[sentinel] Face {i+1} → {name} (score: {best_score:.2f})")
                else:
                    print(f"[sentinel] Face {i+1} → Unknown (best score: {best_score:.2f})")
            else:
                print(f"[sentinel] Face {i+1} → Unknown (no known faces registered)")

            results.append(name)

        return results

    except Exception as e:
        print(f"[sentinel] recognize_faces error: {e}")
        return ["Unknown"]

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({
        "status":      "running",
        "environment": "cloud" if IS_CLOUD else "local",
        "model_ready": FACE_APP_READY
    })

@app.route("/health")
def health():
    return jsonify({
        "status":        "ok",
        "environment":   "cloud" if IS_CLOUD else "local",
        "model_ready":   FACE_APP_READY,
        "known_faces":   len(known_encodings),
    })

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    if not FACE_APP_READY:
        return jsonify({"error": "Face model not ready, try again shortly"}), 503

    file     = request.files["image"]
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    results       = recognize_faces(filepath)
    unknown_count = results.count("Unknown")

    if unknown_count > 0:
        send_telegram_alert(filepath, unknown_count)
        conn = get_db()
        conn.execute(
            "INSERT INTO alerts (timestamp, status, image_path) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), "Unknown", f"uploads/{filename}")
        )
        conn.commit()
        conn.close()
        print(f"[sentinel] {unknown_count} unknown — alert saved")
    else:
        print(f"[sentinel] All known: {results}")

    return jsonify({
        "faces_detected": len(results),
        "results":        results        # ["prasad", "Unknown"] — same as before
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
    Register a new known face.
    POST form-data:
      name  = "prasad"
      image = <photo file>
    """
    global known_encodings

    if "image" not in request.files or "name" not in request.form:
        return jsonify({"error": "Provide 'image' file and 'name' text field"}), 400

    if not FACE_APP_READY:
        return jsonify({"error": "Face model not ready"}), 503

    name     = request.form["name"].strip().replace(" ", "_").lower()
    file     = request.files["image"]
    savepath = os.path.join(KNOWN_DIR, f"{name}.jpg")
    file.save(savepath)

    # Generate and save embedding immediately
    emb = get_face_embedding(savepath)
    if emb is None:
        os.remove(savepath)
        return jsonify({"error": f"No face detected in the uploaded photo for '{name}'"}), 400

    known_encodings[name] = emb
    save_encodings(known_encodings)

    print(f"[sentinel] Registered: {name}")
    return jsonify({
        "message": f"Known face '{name}' registered successfully",
        "total_known": len(known_encodings)
    })

@app.route("/known-faces")
def list_known_faces():
    return jsonify({
        "known_faces": list(known_encodings.keys()),
        "count":       len(known_encodings)
    })

@app.route("/known-faces/<name>", methods=["DELETE"])
def delete_known_face(name):
    global known_encodings
    if name in known_encodings:
        del known_encodings[name]
        save_encodings(known_encodings)
        photo = os.path.join(KNOWN_DIR, f"{name}.jpg")
        if os.path.exists(photo):
            os.remove(photo)
        return jsonify({"message": f"Deleted: {name}"})
    return jsonify({"error": "Not found"}), 404

# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = not IS_CLOUD
    print(f"[sentinel] Starting on port {port}  debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)
