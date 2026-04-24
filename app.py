"""
Brain Tumour Classification – Gradio Web Application
Hosts a MobileNetV2-based model that classifies brain MRI scans
into: Glioma, Meningioma, No Tumour, or Pituitary tumour.

Powered by Gradio for easy local use and one-click Hugging Face Spaces deployment.
"""

import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

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
    print("WARNING: No model file found! The app will start but predictions will fail.")
    print(f"  Looked for: {MODEL_PATHS}")


# ── Prediction Function ─────────────────────────────────────
def predict(image):
    """
    Accepts a PIL Image from Gradio, runs the model, and returns:
      - label_dict:   {display_label: confidence} for gr.Label
      - details_md:   Markdown-formatted detailed results
    """
    if model is None:
        raise gr.Error("⚠️ Model not loaded. Please place a .keras model file in the project directory.")

    if image is None:
        raise gr.Error("⚠️ Please upload an image first.")

    # Preprocess — Gradio gives us a PIL Image
    img = image.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Handle grayscale → 3-channel
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA → RGB
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array, verbose=0)[0]

    # Build results
    results = []
    label_dict = {}
    for i, class_key in enumerate(CLASS_NAMES):
        info = CLASS_DISPLAY[class_key]
        confidence = float(preds[i])
        label_dict[info["label"]] = confidence
        results.append({
            "class": info["label"],
            "icon":  info["icon"],
            "confidence": confidence * 100,
            "color": info["color"],
            "severity": info["severity"],
        })

    # Sort by confidence descending
    results.sort(key=lambda x: x["confidence"], reverse=True)
    top = results[0]

    # Build detailed Markdown output
    severity_emoji = {"High": "🔴", "Medium": "🟡", "None": "🟢"}.get(top["severity"], "⚪")
    details_md = f"""
## {top['icon']} Prediction: **{top['class']}**

| Metric | Value |
|--------|-------|
| **Confidence** | **{top['confidence']:.1f}%** |
| **Severity** | {severity_emoji} {top['severity']} |

---

### All Class Probabilities

| Class | Confidence |
|-------|-----------|
"""
    for r in results:
        bar_len = int(r["confidence"] / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        details_md += f"| {r['icon']} {r['class']} | `{bar}` **{r['confidence']:.1f}%** |\n"

    details_md += """
---
> ⚠️ **Disclaimer:** This is a research/demo tool only and is **NOT** intended for clinical diagnosis.
> Always consult a qualified medical professional for health decisions.
"""

    return label_dict, details_md


# ── About Page Content ───────────────────────────────────────
ABOUT_MD = """
## 🧠 About This Project

This web application uses a deep learning model to classify brain MRI scans into four categories.
It is built for educational and research purposes only.

### Tumour Classes

| Class | Description | Severity |
|-------|------------|----------|
| 🔴 Glioma | Tumour originating from glial cells | High |
| 🟠 Meningioma | Tumour arising from the meninges | Medium |
| 🟡 Pituitary | Tumour in the pituitary gland | Medium |
| 🟢 No Tumour | No tumour detected in scan | None |

### Model Architecture

- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Training Strategy:** Two-phase transfer learning (frozen → fine-tuned)
- **Input Size:** 224 × 224 pixels (RGB)
- **Data Augmentation:** Rotation, shifts, zoom, flips, brightness
- **Framework:** TensorFlow / Keras

### Tech Stack

- **Backend:** Python, Gradio
- **ML Framework:** TensorFlow 2.x (CPU)
- **Deployment:** Hugging Face Spaces compatible

---

> ⚠️ **Disclaimer:** This tool is a **research/demo project** and must **NOT** be used for actual medical diagnosis.
> Always consult a qualified healthcare professional for medical advice.
"""


# ── Build Gradio Interface ───────────────────────────────────
with gr.Blocks(title="Brain Tumour Classifier – AI-Powered MRI Analysis") as demo:

    # Header
    gr.Markdown("# 🧠 Brain Tumour Classifier")
    gr.Markdown(
        "Upload a brain MRI scan and get instant AI-powered classification "
        "using deep learning (MobileNetV2)."
    )

    with gr.Tabs():
        # ── Classify Tab ─────────────────────────────────
        with gr.TabItem("🔬 Classify"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil",
                        label="Upload Brain MRI Scan",
                        sources=["upload", "clipboard"],
                        height=320,
                    )
                    analyse_btn = gr.Button(
                        "🔬 Analyse Scan",
                        variant="primary",
                        size="lg",
                    )

                with gr.Column(scale=1):
                    label_output = gr.Label(
                        label="Classification Result",
                        num_top_classes=4,
                    )

            details_output = gr.Markdown(label="Detailed Results")

            # Wire up prediction
            analyse_btn.click(
                fn=predict,
                inputs=[image_input],
                outputs=[label_output, details_output],
            )
            image_input.change(
                fn=predict,
                inputs=[image_input],
                outputs=[label_output, details_output],
            )

        # ── About Tab ────────────────────────────────────
        with gr.TabItem("ℹ️ About"):
            gr.Markdown(ABOUT_MD)

    # Footer
    gr.Markdown(
        "<center><small>Brain Tumour Classifier · Powered by TensorFlow &amp; "
        "MobileNetV2 · Built with Gradio</small></center>"
    )


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
