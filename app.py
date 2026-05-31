import os, io, base64, cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
app = Flask(__name__)
CORS(app)

# Load model and label map once at startup
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "best.pt")
LABELMAP_PATH = os.path.join(os.path.dirname(__file__), "label_map.txt")

print("Loading model...")
model = YOLO(MODEL_PATH)

with open(LABELMAP_PATH) as f:
    LABELS = {int(l.split(":")[0]): l.split(":")[1].strip() for l in f}
print(f"Model loaded. {len(LABELS)} classes.")

def run_inference(img_bytes):
    """Run YOLOv8 inference on raw image bytes. Returns list of detections."""
    nparr  = np.frombuffer(img_bytes, np.uint8)
    img    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    results = model(img, imgsz=320, conf=0.5, verbose=False)
    boxes   = results[0].boxes
    detections = []
    for box in boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        detections.append({
            "label"     : LABELS.get(cls, str(cls)),
            "confidence": round(conf, 3),
            "bbox"      : {"x1":x1,"y1":y1,"x2":x2,"y2":y2}
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": len(LABELS)})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image field"}), 400
        img_bytes  = request.files["image"].read()
        detections = run_inference(img_bytes)
        top        = detections[0] if detections else None
        return jsonify({
            "success"        : True,
            "detections"     : detections,
            "top_prediction" : top["label"]      if top else None,
            "top_confidence" : top["confidence"] if top else None,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/predict_base64", methods=["POST"])
def predict_base64():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"success": False, "error": "No image field"}), 400
        img_bytes  = base64.b64decode(data["image"])
        detections = run_inference(img_bytes)
        top        = detections[0] if detections else None
        return jsonify({
            "success"        : True,
            "detections"     : detections,
            "top_prediction" : top["label"]      if top else None,
            "top_confidence" : top["confidence"] if top else None,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)