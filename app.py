"""
Brain Tumour Classification – Flask Web Application
Hosts a MobileNetV2-based model that classifies brain MRI scans
into: Glioma, Meningioma, No Tumour, or Pituitary tumour.
"""

import os
import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ── Configuration ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
IMG_SIZE = (224, 224)

# Class labels (must match the training generator's alphabetical order)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

CLASS_DISPLAY = {
    "glioma":      {"label": "Glioma",              "icon": "🔴", "severity": "High",   "color": "#ef4444"},
    "meningioma":  {"label": "Meningioma",           "icon": "🟠", "severity": "Medium", "color": "#f97316"},
    "notumor":     {"label": "No Tumour Detected",   "icon": "🟢", "severity": "None",   "color": "#22c55e"},
    "pituitary":   {"label": "Pituitary Tumour",     "icon": "🟡", "severity": "Medium", "color": "#eab308"},
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
    print("WARNING: No model file found! The app will start but predictions will fail.")
    print(f"  Looked for: {MODEL_PATHS}")

# ── Flask app ────────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(image_path: str) -> dict:
    """Run prediction on a single image and return results."""
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]

    results = []
    for i, class_key in enumerate(CLASS_NAMES):
        info = CLASS_DISPLAY[class_key]
        results.append({
            "class": info["label"],
            "icon": info["icon"],
            "confidence": float(preds[i]) * 100,
            "color": info["color"],
            "severity": info["severity"],
        })

    # Sort by confidence descending
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return {"predictions": results, "top": results[0]}


# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500

    # Save with unique name
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        result = predict(filepath)
        result["image_url"] = f"/static/uploads/{filename}"
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/about")
def about():
    return render_template("about.html")


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
