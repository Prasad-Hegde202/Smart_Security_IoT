from flask import Flask, request, jsonify
import os
import face_recognition
import pickle
import numpy as np
import requests
import sqlite3
from datetime import datetime
from flask import send_from_directory
from flask import render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Folder to store uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔐 Telegram Config (REPLACE THESE)
BOT_TOKEN = "8693553244:AAE4VtpVd2S7HOtk-wdWnbqcBW1no12S50g"
CHAT_ID = "6580212381"

# Load known face encodings
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)


# 🧠 Initialize Database
def init_db():
    conn = sqlite3.connect("alerts.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            status TEXT,
            image_path TEXT
        )
    """)

    conn.commit()
    conn.close()

# Call once at startup
init_db()


# 🚨 Telegram Alert Function
def send_telegram_alert(image_path):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    with open(image_path, "rb") as img:
        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "caption": "🚨 Unknown person detected!"
            },
            files={"photo": img}
        )

    return response.json()


@app.route("/")
def home():
    return "Smart Security Backend Running 🚀"


@app.route("/upload", methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Load image
    image = face_recognition.load_image_file(filepath)

    # Detect faces
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    results = []

    for encoding in face_encodings:
        name = "Unknown"

        if len(data["encodings"]) > 0:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            face_distances = face_recognition.face_distance(data["encodings"], encoding)

            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = data["names"][best_match_index]

        # 🚨 If Unknown → Alert + Save to DB
        if name == "Unknown":
            send_telegram_alert(filepath)

            conn = sqlite3.connect("alerts.db")
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO alerts (timestamp, status, image_path)
                VALUES (?, ?, ?)
            """, (str(datetime.now()), "Unknown", filepath))

            conn.commit()
            conn.close()

        results.append(name)

    return jsonify({
        "faces_detected": len(results),
        "results": results
    })


# 🔍 API to fetch alerts (for dashboard later)
@app.route("/alerts", methods=["GET"])
def get_alerts():
    conn = sqlite3.connect("alerts.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM alerts ORDER BY id DESC")
    rows = cursor.fetchall()

    conn.close()

    alerts_list = []
    for row in rows:
        alerts_list.append({
            "id": row[0],
            "timestamp": row[1],
            "status": row[2],
            "image": row[3]
        })

    return jsonify(alerts_list)

@app.route('/uploads/<filename>')
def serve_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/dashboard")
def dashboard():
    return render_template("index.html")

if __name__ == "__main__":
     port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)