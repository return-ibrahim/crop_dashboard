from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import os
import json
import queue as _queue
import asyncio
import threading

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Gemini ────────────────────────────────────────────────────────────────────
from google import genai as google_genai
from google.genai import types as genai_types

_gemini_client = None
try:
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        _gemini_client = google_genai.Client(api_key=gemini_key)
except Exception as e:
    print(f"⚠️  Gemini init failed: {e}")

# ── YOLO — lazy load ──────────────────────────────────────────────────────────
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

# ── Fallback pesticide DB ─────────────────────────────────────────────────────
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

# ── Drone frame queue (shared between drone_bridge push and WebRTC track) ─────
_frame_queue = _queue.Queue(maxsize=5)

def _make_placeholder_frame():
    """Returns a numpy RGB frame (384×640) shown before the drone connects."""
    h, w = 384, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(frame, "Waiting for E88 drone feed...",
                (60, h // 2 - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 80), 2, cv2.LINE_AA)
    cv2.putText(frame, "Run drone_bridge.py on your laptop",
                (100, h // 2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ── asyncio event loop — runs in a background daemon thread ──────────────────
# Flask is synchronous WSGI; aiortc needs async. We bridge with run_coroutine_threadsafe.
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True, name="aiortc-loop").start()

# Keep references to active peer connections so they can be cleaned up
_pcs: set = set()

# ── WebRTC video track that drains the frame queue ───────────────────────────
try:
    import av
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import VideoStreamTrack
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    print("⚠️  aiortc not installed — WebRTC endpoint will return 501. "
          "Add 'aiortc' and 'av' to requirements.txt and redeploy.")

if AIORTC_AVAILABLE:
    class DroneVideoTrack(VideoStreamTrack):
        """
        Pulls JPEG frames pushed by drone_bridge.py from _frame_queue,
        decodes them to RGB numpy arrays, and wraps them as av.VideoFrame
        objects for aiortc to packetize and send over RTP.
        """
        kind = "video"
        _placeholder = None   # cached so we only build it once

        def __init__(self):
            super().__init__()
            self._last_frame = None

        async def recv(self):
            pts, time_base = await self.next_timestamp()

            # Try to grab the newest frame (non-blocking)
            frame_bytes = None
            while not _frame_queue.empty():
                frame_bytes = _frame_queue.get_nowait()  # drain stale frames

            if frame_bytes is not None:
                nparr = np.frombuffer(frame_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self._last_frame = img

            if self._last_frame is not None:
                img = self._last_frame
            else:
                # No drone connected yet — show placeholder
                if DroneVideoTrack._placeholder is None:
                    DroneVideoTrack._placeholder = _make_placeholder_frame()
                img = DroneVideoTrack._placeholder

            video_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame

    # ── Async handler — creates a PeerConnection and returns the SDP answer ──
    async def _handle_webrtc_offer(offer_sdp: str, offer_type: str) -> dict:
        offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
        pc    = RTCPeerConnection()
        _pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            print(f"WebRTC connection state: {pc.connectionState}")
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                _pcs.discard(pc)

        # Add our drone video track to the outgoing stream
        pc.addTrack(DroneVideoTrack())

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # Wait for ICE gathering to finish so the SDP contains all candidates
        # (avoids the need for trickle-ICE on the client)
        gather_complete = asyncio.Event()

        @pc.on("icegatheringstatechange")
        def on_ice_gathering():
            if pc.iceGatheringState == "complete":
                gather_complete.set()

        if pc.iceGatheringState != "complete":
            try:
                await asyncio.wait_for(gather_complete.wait(), timeout=10)
            except asyncio.TimeoutError:
                pass   # send whatever ICE candidates we have

        return {
            "sdp":  pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health_check():
    model_ok = get_yolo() is not None
    return jsonify({
        "status": "online",
        "yolo":   "active" if model_ok else "missing — upload backend/models/best.pt to repo",
        "gemini": "connected" if _gemini_client else "not configured — set GEMINI_API_KEY",
        "webrtc": "available" if AIORTC_AVAILABLE else "unavailable — install aiortc",
    }), 200


@app.route("/api/webrtc/offer", methods=["POST"])
def webrtc_offer():
    """
    Browser sends:  { sdp: "...", type: "offer" }
    We return:      { sdp: "...", type: "answer" }

    The browser then calls pc.setRemoteDescription(answer) and the
    DroneVideoTrack starts flowing over RTP/SRTP.
    """
    if not AIORTC_AVAILABLE:
        return jsonify({"error": "aiortc not installed on server"}), 501

    data = request.get_json(force=True)
    if not data or "sdp" not in data:
        return jsonify({"error": "Missing SDP in request body"}), 400

    # Bridge the async coroutine into our background event loop
    future = asyncio.run_coroutine_threadsafe(
        _handle_webrtc_offer(data["sdp"], data.get("type", "offer")),
        _loop,
    )
    try:
        result = future.result(timeout=30)
        return jsonify(result)
    except Exception as e:
        print(f"WebRTC offer error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/drone_frame", methods=["POST"])
def receive_drone_frame():
    """Receives JPEG frames pushed by drone_bridge.py running on the laptop."""
    frame_bytes = request.data
    if not frame_bytes:
        return jsonify({"error": "No frame data"}), 400

    # Keep the queue from growing stale — drop oldest if full
    if _frame_queue.full():
        try:
            _frame_queue.get_nowait()
        except Exception:
            pass
    _frame_queue.put(frame_bytes)
    return jsonify({"status": "ok"}), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    """Run YOLO inference on a posted image and return disease + AI treatment."""
    model = get_yolo()
    if model is None:
        return jsonify({"error": "YOLO model not loaded — upload best.pt to backend/models/best.pt"}), 503
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        file    = request.files["image"].read()
        image   = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
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