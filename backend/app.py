from flask import Flask, request, jsonify, send_from_directory
import os
import requests
import sqlite3
import numpy as np
import cv2
import pickle
import base64
from datetime import datetime
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# ── Environment ────────────────────────────────────────────────────────────
IS_CLOUD    = bool(os.environ.get("RENDER", False))
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000").rstrip("/")

if IS_CLOUD:
    BASE_DIR = os.environ.get("DATA_DIR", "/tmp/sentinel")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# uploads/ still used locally and for Telegram sending (temp file)
UPLOAD_DIR     = os.path.join(BASE_DIR, "uploads")
DB_PATH        = os.path.join(BASE_DIR, "alerts.db")
KNOWN_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_faces")
ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pkl")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(KNOWN_DIR,  exist_ok=True)

print(f"[sentinel] Mode       : {'CLOUD' if IS_CLOUD else 'LOCAL'}")
print(f"[sentinel] Backend URL: {BACKEND_URL}")
print(f"[sentinel] Database   : {DB_PATH}")

# ── Telegram ───────────────────────────────────────────────────────────────
BOT_TOKEN = os.environ.get("BOT_TOKEN", "8693553244:AAE4VtpVd2S7HOtk-wdWnbqcBW1no12S50g")
CHAT_ID   = os.environ.get("CHAT_ID",   "6580212381")

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
                    "chat_id":    CHAT_ID,
                    "caption":    f"🚨 Unknown person detected!\n👤 Count: {unknown_count}\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "parse_mode": "Markdown"
                },
                files={"photo": img},
                timeout=15
            )
        print(f"[sentinel] Telegram sent: {resp.status_code}")
        if resp.status_code != 200:
            print(f"[sentinel] Telegram error body: {resp.text}")
    except Exception as e:
        print(f"[sentinel] Telegram error: {e}")

# ── Database ───────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    # image_data stores base64 encoded image — survives server restarts
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT NOT NULL,
            status     TEXT NOT NULL,
            image_data TEXT,
            image_path TEXT
        )
    """)
    # migrate old schema if needed
    try:
        conn.execute("ALTER TABLE alerts ADD COLUMN image_data TEXT")
    except Exception:
        pass
    conn.commit()
    conn.close()

init_db()

# ── InsightFace ────────────────────────────────────────────────────────────
print("[sentinel] Loading InsightFace...")
try:
    from insightface.app import FaceAnalysis
    face_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(480, 480))
    print("[sentinel] InsightFace ready ✓")
    FACE_APP_READY = True
except Exception as e:
    print(f"[sentinel] InsightFace failed: {e}")
    face_app       = None
    FACE_APP_READY = False

# ── Known encodings ────────────────────────────────────────────────────────
def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_encodings(enc):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(enc, f)

def get_embedding(image_path):
    if not FACE_APP_READY:
        return None
    try:
        img   = cv2.imread(image_path)
        faces = face_app.get(img)
        if not faces:
            return None
        largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        return largest.embedding
    except Exception as e:
        print(f"[sentinel] Embedding error: {e}")
        return None

def rebuild_from_folder():
    enc    = {}
    photos = [f for f in os.listdir(KNOWN_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))]
    for photo in photos:
        name = os.path.splitext(photo)[0]
        emb  = get_embedding(os.path.join(KNOWN_DIR, photo))
        if emb is not None:
            enc[name] = emb
            print(f"[sentinel] Encoded: {name}")
        else:
            print(f"[sentinel] No face in {photo} — skipped")
    save_encodings(enc)
    return enc

known_encodings = load_encodings()
print(f"[sentinel] Known faces loaded: {len(known_encodings)}")

if not known_encodings:
    photos = [f for f in os.listdir(KNOWN_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))]
    if photos:
        print("[sentinel] Rebuilding encodings from known_faces/ ...")
        known_encodings = rebuild_from_folder()

# ── Face recognition ───────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.35

def recognize_faces(image_path):
    if not FACE_APP_READY:
        return ["Unknown"]
    try:
        img  = cv2.imread(image_path)
        h, w = img.shape[:2]
        if w < 640:
            scale = 640 / w
            img   = cv2.resize(img, (int(w*scale), int(h*scale)))

        faces = face_app.get(img)
        if not faces:
            print("[sentinel] No faces detected")
            return []

        print(f"[sentinel] {len(faces)} face(s) detected")
        results = []

        for i, face in enumerate(faces):
            name = "Unknown"
            if known_encodings:
                q = face.embedding.reshape(1, -1)
                best_name, best_score = None, -1
                for kname, kemb in known_encodings.items():
                    score = cosine_similarity(q, kemb.reshape(1, -1))[0][0]
                    if score > best_score:
                        best_score = score
                        best_name  = kname
                if best_score >= SIMILARITY_THRESHOLD:
                    name = best_name
                    print(f"[sentinel] Face {i+1} → {name} ({best_score:.2f})")
                else:
                    print(f"[sentinel] Face {i+1} → Unknown ({best_score:.2f})")
            else:
                print(f"[sentinel] Face {i+1} → Unknown (no known faces)")
            results.append(name)

        return results
    except Exception as e:
        print(f"[sentinel] recognize_faces error: {e}")
        return ["Unknown"]

# ── Helper: image file → base64 data URI ──────────────────────────────────
def file_to_base64(filepath):
    """Convert image file to base64 data URI for storing in DB."""
    try:
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = filepath.rsplit(".", 1)[-1].lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"[sentinel] base64 encode error: {e}")
        return None

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({
        "status":      "running",
        "environment": "cloud" if IS_CLOUD else "local",
        "model_ready": FACE_APP_READY,
        "known_faces": len(known_encodings)
    })

@app.route("/health")
def health():
    return jsonify({
        "status":      "ok",
        "environment": "cloud" if IS_CLOUD else "local",
        "model_ready": FACE_APP_READY,
        "known_faces": len(known_encodings),
        "backend_url": BACKEND_URL
    })

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    if not FACE_APP_READY:
        return jsonify({"error": "Model not ready, try again shortly"}), 503

    file     = request.files["image"]
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    results       = recognize_faces(filepath)
    unknown_count = results.count("Unknown")

    if unknown_count > 0:
        # Send Telegram with the actual file (still on disk right now)
        send_telegram_alert(filepath, unknown_count)

        # Store image as base64 in DB — persists forever, survives restarts
        image_data = file_to_base64(filepath)

        conn = get_db()
        conn.execute(
            "INSERT INTO alerts (timestamp, status, image_data, image_path) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), "Unknown", image_data, f"uploads/{filename}")
        )
        conn.commit()
        conn.close()
        print(f"[sentinel] Alert saved with base64 image — {unknown_count} unknown")
    else:
        print(f"[sentinel] All known: {results}")

    # Clean up temp file on cloud to save /tmp space
    if IS_CLOUD and os.path.exists(filepath):
        os.remove(filepath)

    return jsonify({
        "faces_detected": len(results),
        "results":        results
    })

@app.route("/alerts")
def get_alerts():
    conn = get_db()
    rows = conn.execute("SELECT * FROM alerts ORDER BY id DESC").fetchall()
    conn.close()

    alerts = []
    for r in rows:
        d = dict(r)
        # image field: use base64 data URI if available, else fallback to URL
        if d.get("image_data"):
            d["image"] = d["image_data"]   # base64 data URI — works anywhere
        elif d.get("image_path"):
            d["image"] = f"{BACKEND_URL}/{d['image_path']}"
        else:
            d["image"] = ""
        # don't send raw image_data in listing (too large, image field has it)
        del d["image_data"]
        alerts.append(d)

    return jsonify(alerts)

@app.route("/uploads/<filename>")
def serve_image(filename):
    # Only works locally; on cloud files are deleted after processing
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/alerts/clear", methods=["DELETE"])
def clear_alerts():
    conn = get_db()
    conn.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()
    return jsonify({"message": "All alerts cleared"})

# ── Known face management ──────────────────────────────────────────────────
@app.route("/known-faces/add", methods=["POST"])
def add_known_face():
    global known_encodings
    if "image" not in request.files or "name" not in request.form:
        return jsonify({"error": "Provide 'image' file and 'name' field"}), 400
    if not FACE_APP_READY:
        return jsonify({"error": "Model not ready"}), 503

    name     = request.form["name"].strip().replace(" ", "_").lower()
    file     = request.files["image"]
    savepath = os.path.join(KNOWN_DIR, f"{name}.jpg")
    file.save(savepath)

    emb = get_embedding(savepath)
    if emb is None:
        os.remove(savepath)
        return jsonify({"error": f"No face detected in photo for '{name}'. Use a clear front-facing photo."}), 400

    known_encodings[name] = emb
    save_encodings(known_encodings)
    print(f"[sentinel] Registered: {name}")
    return jsonify({"message": f"'{name}' registered", "total_known": len(known_encodings)})

@app.route("/known-faces")
def list_known_faces():
    return jsonify({"known_faces": list(known_encodings.keys()), "count": len(known_encodings)})

@app.route("/known-faces/<n>", methods=["DELETE"])
def delete_known_face(n):
    global known_encodings
    if n in known_encodings:
        del known_encodings[n]
        save_encodings(known_encodings)
        photo = os.path.join(KNOWN_DIR, f"{n}.jpg")
        if os.path.exists(photo):
            os.remove(photo)
        return jsonify({"message": f"Deleted: {n}"})
    return jsonify({"error": "Not found"}), 404

# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = not IS_CLOUD
    print(f"[sentinel] Starting on port {port}  debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)
