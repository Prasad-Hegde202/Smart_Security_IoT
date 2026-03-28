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
IS_CLOUD   = bool(os.environ.get("RENDER", False))
# YOUR Render backend URL — used to build absolute image URLs for dashboard
# Set this as an env var on Render: BACKEND_URL=https://smart-security-backend-tvks.onrender.com
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000").rstrip("/")

if IS_CLOUD:
    BASE_DIR = os.environ.get("DATA_DIR", "/tmp/sentinel")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR     = os.path.join(BASE_DIR, "uploads")
DB_PATH        = os.path.join(BASE_DIR, "alerts.db")
KNOWN_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_faces")
ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pkl")  # in BASE_DIR so it persists in /tmp

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(KNOWN_DIR,  exist_ok=True)

print(f"[sentinel] Mode       : {'CLOUD' if IS_CLOUD else 'LOCAL'}")
print(f"[sentinel] Backend URL: {BACKEND_URL}")
print(f"[sentinel] Uploads    : {UPLOAD_DIR}")
print(f"[sentinel] Database   : {DB_PATH}")

# ── Telegram ───────────────────────────────────────────────────────────────
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID   = os.environ.get("CHAT_ID",   "")

def send_telegram_alert(image_path, unknown_count, image_url):
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
                    "caption": (
                        f"🚨 *Unknown person detected!*\n"
                        f"👤 Unknown faces: {unknown_count}\n"
                        f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"🔗 {image_url}"
                    ),
                    "parse_mode": "Markdown"
                },
                files={"photo": img},
                timeout=15
            )
        print(f"[sentinel] Telegram sent: {resp.status_code}")
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
            image_path TEXT NOT NULL,
            image_url  TEXT
        )
    """)
    # add image_url column if upgrading from old schema
    try:
        conn.execute("ALTER TABLE alerts ADD COLUMN image_url TEXT")
    except Exception:
        pass
    conn.commit()
    conn.close()

init_db()

# ── InsightFace setup ──────────────────────────────────────────────────────
print("[sentinel] Loading InsightFace...")
try:
    from insightface.app import FaceAnalysis

    face_app = FaceAnalysis(
        name      = "buffalo_sc",
        providers = ["CPUExecutionProvider"]
    )
    # FIX 2: larger det_size catches more faces (480 is a good balance for free tier)
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
    """Get embedding of the largest face in an image."""
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
    """Build encodings from all photos in known_faces/ folder."""
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

# Rebuild from folder if encodings empty but photos exist
if not known_encodings:
    photos = [f for f in os.listdir(KNOWN_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))]
    if photos:
        print("[sentinel] Rebuilding encodings from known_faces/ ...")
        known_encodings = rebuild_from_folder()

# ── Face recognition ───────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.35  # lowered slightly for better recall

def recognize_faces(image_path):
    """
    Returns list of names for every face in image.
    e.g. ["prasad", "Unknown"]
    """
    if not FACE_APP_READY:
        return ["Unknown"]
    try:
        img   = cv2.imread(image_path)
        # FIX 2: upscale small images so small faces are detected
        h, w  = img.shape[:2]
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
                q     = face.embedding.reshape(1, -1)
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

    file      = request.files["image"]
    filename  = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath  = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    # FIX 1: absolute URL so dashboard loads images correctly from cloud
    image_url = f"{BACKEND_URL}/uploads/{filename}"

    results       = recognize_faces(filepath)
    unknown_count = results.count("Unknown")

    if unknown_count > 0:
        # FIX 4: pass filepath (not URL) to open the file, pass URL for caption
        send_telegram_alert(filepath, unknown_count, image_url)

        conn = get_db()
        conn.execute(
            "INSERT INTO alerts (timestamp, status, image_path, image_url) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), "Unknown", f"uploads/{filename}", image_url)
        )
        conn.commit()
        conn.close()
        print(f"[sentinel] Alert saved — {unknown_count} unknown")
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

    alerts = []
    for r in rows:
        d = dict(r)
        # FIX 1: always return absolute image URL to dashboard
        # if image_url stored → use it; else build it from image_path
        if d.get("image_url"):
            d["image"] = d["image_url"]
        else:
            d["image"] = f"{BACKEND_URL}/{d['image_path']}"
        alerts.append(d)

    return jsonify(alerts)

@app.route("/uploads/<filename>")
def serve_image(filename):
    return send_from_directory(UPLOAD_DIR, filename)

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
    return jsonify({
        "message":     f"'{name}' registered successfully",
        "total_known": len(known_encodings)
    })

@app.route("/known-faces")
def list_known_faces():
    return jsonify({
        "known_faces": list(known_encodings.keys()),
        "count":       len(known_encodings)
    })

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

@app.route("/alerts/clear", methods=["DELETE"])
def clear_alerts():
    conn = get_db()
    conn.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()
    return jsonify({"message": "All alerts cleared"})
```

After clearing, all **new** uploads will have the correct Render URL and images will load in dashboard and Telegram.

---

### Quick checklist to verify it's working

**Step 1** — Check env var is set:
```
GET https://smart-security-backend-tvks.onrender.com/health
# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = not IS_CLOUD
    print(f"[sentinel] Starting on port {port}  debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)
