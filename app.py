"""
Brain Tumour Classification – Flask Web Application
Hosts a MobileNetV2-based model that classifies brain MRI scans
into: Glioma, Meningioma, No Tumour, or Pituitary tumour.

Uses a modern, dark-mode glassmorphism UI.
"""

import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# ── Configuration ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = (224, 224)

# Class labels (must match the training generator's alphabetical order)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

CLASS_DISPLAY = {
    "glioma":     {"label": "Glioma",            "icon": "🔴", "severity": "High",   "color": "#ef4444"},
    "meningioma": {"label": "Meningioma",         "icon": "🟠", "severity": "Medium", "color": "#f97316"},
    "notumor":    {"label": "No Tumour Detected", "icon": "🟢", "severity": "None",   "color": "#22c55e"},
    "pituitary":  {"label": "Pituitary Tumour",   "icon": "🟡", "severity": "Medium", "color": "#eab308"},
}

# ── Model loading ────────────────────────────────────────────
MODEL_PATHS = [
    os.path.join(BASE_DIR, "best_model_final.keras"),
    os.path.join(BASE_DIR, "brain_tumour_model_final.keras"),
    os.path.join(BASE_DIR, "best_model_phase1.keras"),
    os.path.join(BASE_DIR, "my_model.keras"),
]

model = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        print(f"Loading model from: {path}")
        model = load_model(path, compile=False)
        print("Model loaded successfully!")
        break

if model is None:
    print("WARNING: No model file found! Predictions will fail.")
    print(f"  Looked for: {MODEL_PATHS}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts an image file via POST, runs the model, and returns JSON.
    """
    if model is None:
        return jsonify({"error": "Model not loaded on server. Please place a .keras model file in the project directory."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        # Preprocess — Read the image
        img = Image.open(file.stream).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0

        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array, verbose=0)[0]

        # Build results
        results = []
        for i, class_key in enumerate(CLASS_NAMES):
            info = CLASS_DISPLAY[class_key]
            confidence = float(preds[i]) * 100
            results.append({
                "class": info["label"],
                "icon":  info["icon"],
                "confidence": confidence,
                "color": info["color"],
                "severity": info["severity"],
            })

        # Sort by confidence descending
        results.sort(key=lambda x: x["confidence"], reverse=True)
        top = results[0]

        return jsonify({
            "top": top,
            "predictions": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
