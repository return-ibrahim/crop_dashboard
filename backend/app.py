from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import os
import json
import queue as _queue

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Gemini — NEW google.genai SDK (old google.generativeai is fully deprecated) ──
from google import genai as google_genai
from google.genai import types as genai_types

_gemini_client = None
try:
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        _gemini_client = google_genai.Client(api_key=gemini_key)
except Exception as e:
    print(f"⚠️  Gemini init failed: {e}")

# ── YOLO — lazy load so server starts even if best.pt is missing ─────────────
# Upload best.pt to:  backend/models/best.pt  in your GitHub repo
_yolo_model = None

def get_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    from ultralytics import YOLO
    model_path = os.path.join(os.path.dirname(__file__), "models", "best.pt")
    if not os.path.exists(model_path):
        print(f"❌  best.pt not found at {model_path}. YOLO predictions disabled.")
        return None
    _yolo_model = YOLO(model_path)
    print(f"✅  YOLO model loaded from {model_path}")
    return _yolo_model

# ── Fallback pesticide DB ────────────────────────────────────────────────────
PESTICIDE_DB = {
    "Rice Blast":       {"pesticide": "Tricyclazole 75% WP", "rate": "0.6 g/L",      "instructions": "Spray at 10-12 day intervals. Apply at first sign of infection."},
    "Brown Spot":       {"pesticide": "Mancozeb 75% WP",     "rate": "2.0 g/L",      "instructions": "Seed treatment recommended. Repeat after 14 days if needed."},
    "Bacterial Blight": {"pesticide": "Streptocycline",       "rate": "300g/hectare", "instructions": "Spray at first appearance. Drain fields to reduce spread."},
    "Healthy":          {"pesticide": "None",                 "rate": "N/A",          "instructions": "Maintain monitoring. No treatment required."},
}

def get_ai_treatment(disease_name):
    if _gemini_client is None or disease_name == "Healthy":
        return PESTICIDE_DB.get(disease_name, PESTICIDE_DB["Healthy"])
    try:
        prompt = (
            f"You are an expert agronomist. A crop has been diagnosed with {disease_name}. "
            "Return ONLY a raw JSON object with exactly three keys: "
            "'pesticide' (recommended chemical name), 'rate' (application rate), and "
            "'instructions' (2 short actionable sentences). No markdown, no extra text."
        )
        response = _gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(max_output_tokens=150, temperature=0.2),
        )
        raw = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(raw)
    except Exception as e:
        print(f"⚠️  Gemini treatment failed, using fallback. Error: {e}")
        return PESTICIDE_DB.get(disease_name, PESTICIDE_DB["Healthy"])

# ── Drone frame queue ────────────────────────────────────────────────────────
_frame_queue = _queue.Queue(maxsize=3)

def _make_placeholder():
    h, w = 384, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(frame, "Waiting for E88 drone feed...",
                (60, h // 2 - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 80), 2, cv2.LINE_AA)
    cv2.putText(frame, "Run drone_bridge.py on your laptop",
                (100, h // 2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()

def generate_frames():
    placeholder = _make_placeholder()
    while True:
        try:
            frame_bytes = _frame_queue.get(timeout=3)
        except _queue.Empty:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health_check():
    model_ok = get_yolo() is not None
    return jsonify({
        "status": "online",
        "yolo":   "active" if model_ok else "missing — upload backend/models/best.pt to repo",
        "gemini": "connected" if _gemini_client else "not configured — set GEMINI_API_KEY",
    }), 200

@app.route("/video_feed", methods=["GET"])
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/drone_frame", methods=["POST"])
def receive_drone_frame():
    frame_bytes = request.data
    if not frame_bytes:
        return jsonify({"error": "No frame data"}), 400
    if _frame_queue.full():
        try: _frame_queue.get_nowait()
        except: pass
    _frame_queue.put(frame_bytes)
    return jsonify({"status": "ok"}), 200

@app.route("/api/predict", methods=["POST"])
def predict():
    model = get_yolo()
    if model is None:
        return jsonify({"error": "YOLO model not loaded — upload best.pt to backend/models/best.pt"}), 503
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        file  = request.files["image"].read()
        image = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
        results = model.predict(image, conf=0.5)[0]
        if len(results.boxes) == 0:
            diagnosis, conf = "Healthy", 100.0
        else:
            cls_id    = int(results.boxes[0].cls[0])
            diagnosis = model.names[cls_id]
            conf      = round(float(results.boxes[0].conf[0]) * 100, 2)
        treatment = get_ai_treatment(diagnosis)
        return jsonify({
            "disease":      diagnosis,
            "confidence":   conf,
            "pesticide":    treatment.get("pesticide",    "Consult Agronomist"),
            "rate":         treatment.get("rate",         "N/A"),
            "instructions": treatment.get("instructions", "No specific instructions available."),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)