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
def get_ai_treatment(disease_name):
    """Fetches a dynamic treatment plan from Gemini (free tier)."""
    if _gemini_model is None or disease_name == "Healthy":
        return PESTICIDE_DB.get(disease_name, PESTICIDE_DB["Healthy"])

    try:
        prompt = (
            f"You are an expert agronomist. A crop has been diagnosed with {disease_name}. "
            "Return ONLY a raw JSON object with exactly three keys: "
            "'pesticide' (recommended chemical name), 'rate' (application rate), and "
            "'instructions' (2 short sentences of actionable advice). "
            "Do not include markdown formatting or any other text."
        )
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
        print(f"⚠️  Gemini treatment failed, using fallback. Error: {e}")
        return PESTICIDE_DB.get(disease_name, PESTICIDE_DB["Healthy"])


# ── Video-stream globals ────────────────────────────────────────────────────
_camera     = None
_camera_lock = threading.Lock()

def get_camera():
    """Return a shared VideoCapture, opening it lazily."""
    global _camera
    with _camera_lock:
        if _camera is None or not _camera.isOpened():
            # Try index 0 first; on cloud/Render there may be no camera,
            # in which case we return None and the route will send a placeholder.
            _camera = cv2.VideoCapture(0)
        return _camera if _camera.isOpened() else None


def generate_frames():
    """MJPEG generator — overlays YOLO detections on each frame."""
    cap = get_camera()

    if cap is None:
        # No real camera — send a single "no signal" placeholder frame
        h, w = 480, 640
        while True:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(frame, "No Camera / Drone Signal",
                        (80, h // 2 - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 200, 80), 2, cv2.LINE_AA)
            cv2.putText(frame, "Connect camera or drone to enable live feed",
                        (60, h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (100, 100, 100), 1, cv2.LINE_AA)
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')
            # ~1 fps for the placeholder to save bandwidth
            import time; time.sleep(1)
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO detection (low conf threshold for the live feed)
        results = model.predict(frame, conf=0.4, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = f"{model.names[cls_id]} {conf:.0%}"

            # Red boxes for diseases, green for healthy
            color = (0, 255, 0) if model.names[cls_id].lower() == "healthy" else (0, 60, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # Encode and yield
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')


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