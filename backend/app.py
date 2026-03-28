from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
import anthropic
import json

app = Flask(__name__)

# Allow your Vercel frontend to talk to this backend
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best.pt')
model = YOLO(MODEL_PATH)

# Initialize Anthropic Client (It will automatically look for ANTHROPIC_API_KEY in environment variables)
try:
    anthropic_client = anthropic.Anthropic()
except:
    anthropic_client = None

# Fallback Database (Used if AI generation fails)
PESTICIDE_DB = {
    "Rice Blast": {"pesticide": "Tricyclazole 75% WP", "rate": "0.6 g/L", "instructions": "Spray at 10-12 day intervals."},
    "Brown Spot": {"pesticide": "Mancozeb 75% WP", "rate": "2.0 g/L", "instructions": "Seed treatment recommended."},
    "Bacterial Blight": {"pesticide": "Streptocycline", "rate": "300g/hectare", "instructions": "Spray at first appearance."},
    "Healthy": {"pesticide": "None", "rate": "N/A", "instructions": "Maintain monitoring."}
}

def get_ai_treatment(disease_name):
    """Fetches a dynamic treatment plan from Claude AI"""
    if not anthropic_client or disease_name == "Healthy":
        return PESTICIDE_DB.get(disease_name, PESTICIDE_DB["Healthy"])
        
    try:
        prompt = f"You are an expert agronomist. A crop has been diagnosed with {disease_name}. Return ONLY a raw JSON object with exactly three keys: 'pesticide' (recommended chemical name), 'rate' (application rate), and 'instructions' (2 short sentences of actionable advice). Do not include markdown formatting or any other text."
        
        message = anthropic_client.messages.create(
            model="claude-3-haiku-20240307", # Using Haiku for the fastest response time
            max_tokens=150,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse Claude's response into a Python dictionary
        ai_response = json.loads(message.content[0].text)
        return ai_response
    except Exception as e:
        print(f"⚠️ AI Generation Failed, using fallback. Error: {e}")
        return PESTICIDE_DB.get(disease_name, PESTICIDE_DB["Healthy"])

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "model": "YOLOv8 Active"}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image'].read()
        image = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
        
        # 1. Run YOLOv8 Vision Model
        results = model.predict(image, conf=0.5)[0]
        
        if len(results.boxes) == 0:
            diagnosis = "Healthy"
            conf = 100.0
        else:
            class_id = int(results.boxes[0].cls[0])
            diagnosis = model.names[class_id]
            conf = round(float(results.boxes[0].conf[0]) * 100, 2)

        # 2. Get Treatment Plan (From Claude AI or Fallback DB)
        treatment = get_ai_treatment(diagnosis)

        # 3. Send combined package back to the frontend
        return jsonify({
            "disease": diagnosis,
            "confidence": conf,
            "pesticide": treatment.get("pesticide", "Consult Agronomist"),
            "rate": treatment.get("rate", "N/A"),
            "instructions": treatment.get("instructions", "No specific instructions available.")
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)