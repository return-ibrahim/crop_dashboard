from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

# IMPORTANT: CORS allows your Vercel/Netlify frontend to talk to this Render backend.
# For production, replace "*" with your actual frontend URL (e.g., "https://cropguard.vercel.app")
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best.pt')
model = YOLO(MODEL_PATH)

PESTICIDE_DB = {
    "Rice Blast": {"pesticide": "Tricyclazole 75% WP", "rate": "0.6 g/L", "instructions": "Spray at 10-12 day intervals."},
    "Brown Spot": {"pesticide": "Mancozeb 75% WP", "rate": "2.0 g/L", "instructions": "Seed treatment recommended."},
    "Bacterial Blight": {"pesticide": "Streptocycline", "rate": "300g/hectare", "instructions": "Spray at first appearance."},
    "Healthy": {"pesticide": "None", "rate": "N/A", "instructions": "Maintain monitoring."}
}

# 1. API Health Check Route
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "model": "YOLOv8 Active"}), 200

# 2. Prediction Endpoint
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image'].read()
        image = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
        
        results = model.predict(image, conf=0.5)[0]
        
        if len(results.boxes) == 0:
            diagnosis = "Healthy"
            conf = 100.0
        else:
            class_id = int(results.boxes[0].cls[0])
            diagnosis = model.names[class_id]
            conf = round(float(results.boxes[0].conf[0]) * 100, 2)

        treatment = PESTICIDE_DB.get(diagnosis, PESTICIDE_DB["Healthy"])
        return jsonify({
            "disease": diagnosis,
            "confidence": conf,
            "pesticide": treatment["pesticide"],
            "rate": treatment["rate"],
            "instructions": treatment["instructions"]
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)