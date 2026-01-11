from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# -------- LOAD MODEL --------
model = load_model("model/dry_wet_model.h5")

# IMPORTANT: order must match training folder order
classes = ["Dry Waste", "Wet Waste"]

IMG_SIZE = 224

# -------- ROUTES --------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Check image exists
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]

    # Read image from request (NO NEED TO SAVE)
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"})

    # Preprocess image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)[0]
    idx = np.argmax(preds)

    result = classes[idx]
    confidence = round(float(preds[idx]) * 100, 2)

    # DEBUG (optional – you can remove later)
    print("Prediction:", result, confidence)

    # Send response to frontend
    return jsonify({
        "result": result,
        "confidence": confidence
    })

# -------- RUN SERVER --------
if __name__ == "__main__":
    app.run(debug=True)
