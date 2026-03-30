from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
import google.generativeai as genai
import threading

app = Flask(__name__)

# Allow your Vercel frontend to talk to this backend
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best.pt')
model = YOLO(MODEL_PATH)

# Initialize Gemini Client (free tier — get key at https://aistudio.google.com/apikey)
_gemini_model = None
try:
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        genai.configure(api_key=gemini_key)
        _gemini_model = genai.GenerativeModel("gemini-2.0-flash")
except Exception:
    _gemini_model = None

# ── Fallback DB ────────────────────────────────────────────────────────────
PESTICIDE_DB = {
    "Rice Blast":       {"pesticide": "Tricyclazole 75% WP",  "rate": "0.6 g/L",        "instructions": "Spray at 10-12 day intervals. Apply at first sign of infection."},
    "Brown Spot":       {"pesticide": "Mancozeb 75% WP",      "rate": "2.0 g/L",        "instructions": "Seed treatment recommended. Repeat after 14 days if needed."},
    "Bacterial Blight": {"pesticide": "Streptocycline",        "rate": "300g/hectare",   "instructions": "Spray at first appearance. Drain fields to reduce spread."},
    "Healthy":          {"pesticide": "None",                  "rate": "N/A",            "instructions": "Maintain monitoring. No treatment required."},
}


# ── AI Treatment Plan ──────────────────────────────────────────────────────
def get_ai_treatment(disease_name, retries=4, base_delay=2):
    """Fetches a dynamic treatment plan from Gemini, with retry on 429."""
    import time

    if _gemini_model is None or disease_name == "Healthy":
        return PESTICIDE_DB.get(disease_name, PESTICIDE_DB["Healthy"])

    prompt = (
        f"You are an expert agronomist. A crop has been diagnosed with {disease_name}. "
        "Return ONLY a raw JSON object with exactly three keys: "
        "'pesticide' (recommended chemical name), 'rate' (application rate), and "
        "'instructions' (2 short sentences of actionable advice). "
        "Do not include markdown formatting or any other text."
    )

    for attempt in range(retries):
        try:
            response = _gemini_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=150,
                    temperature=0.2,
                )
            )
            raw = response.text.strip().replace("```json", "").replace("```", "")
            return json.loads(raw)

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                wait = base_delay * (2 ** attempt)   # 2s → 4s → 8s → 16s
                print(f"⏳ Gemini 429 – waiting {wait}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
            else:
                print(f"⚠️  Gemini treatment failed, using fallback. Error: {e}")
                break

    # All retries exhausted → use fallback DB silently
    return PESTICIDE_DB.get(disease_name, PESTICIDE_DB["Healthy"])


import queue as _queue

# ── Frame queue — receives JPEG frames pushed from drone_bridge.py ──────────
# E88 drone streams RTSP at rtsp://192.168.1.1:7070/webcam on its local WiFi.
# drone_bridge.py (runs on your laptop) captures that stream and pushes each
# YOLO-annotated frame to /api/drone_frame on this Render backend.
_frame_queue = _queue.Queue(maxsize=3)   # keep only latest 3 frames


def _make_placeholder(msg="Waiting for E88 drone feed…"):
    h, w = 384, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(frame, msg,
                (60, h // 2 - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 200, 80), 2, cv2.LINE_AA)
    cv2.putText(frame, "Run drone_bridge.py on your laptop",
                (100, h // 2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (80, 80, 80), 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


def generate_frames():
    """MJPEG generator — serves YOLO-annotated frames from the E88 drone."""
    import time
    placeholder = _make_placeholder()
    while True:
        try:
            frame_bytes = _frame_queue.get(timeout=3)
        except _queue.Empty:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + placeholder + b"\r\n")
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + frame_bytes + b"\r\n")

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "model": "YOLOv8 Active"}), 200


@app.route('/video_feed', methods=['GET'])
def video_feed():
    """MJPEG stream with live YOLO inference overlay."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/drone_frame', methods=['POST'])
def receive_drone_frame():
    """
    Receives a YOLO-annotated JPEG frame pushed from drone_bridge.py.
    The bridge script runs on your laptop connected to the E88 WiFi hotspot,
    reads rtsp://192.168.1.1:7070/webcam, runs YOLO locally, and POSTs
    each processed frame as raw JPEG bytes to this endpoint.
    """
    frame_bytes = request.data
    if not frame_bytes:
        return jsonify({"error": "No frame data"}), 400

    # Drop oldest frame if queue is full, then enqueue new one
    if _frame_queue.full():
        try:
            _frame_queue.get_nowait()
        except Exception:
            pass
    _frame_queue.put(frame_bytes)
    return jsonify({"status": "ok"}), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    """Single-image YOLO prediction + AI treatment recommendation."""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file  = request.files['image'].read()
        image = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)

        results = model.predict(image, conf=0.5)[0]

        if len(results.boxes) == 0:
            diagnosis = "Healthy"
            conf      = 100.0
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


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)