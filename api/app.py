from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# تحميل الموديل مرة واحدة
model = tf.keras.models.load_model("saved_models/final_model_balanced2.h5")

class_names = ["diseases", "healthy"]

def preprocess_image(img):
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32)  # مهم
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "images" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    files = request.files.getlist("images")

    images = []
    for file in files:
        img = Image.open(file).convert("RGB")
        images.append(preprocess_image(img))

    # shape → (batch, 128, 128, 3)
    images = np.array(images)

    predictions = model.predict(images)

    results = []
    all_confidences = []
    diseases_count = 0

    for i, pred in enumerate(predictions):
        pred_value = float(pred[0])  # حل float32

        healthy_prob = pred_value
        diseases_prob = 1 - pred_value

        if pred_value > 0.5:
            result = "healthy"
            confidence = healthy_prob
        else:
            result = "diseases"
            confidence = diseases_prob
            diseases_count += 1

        confidence_percent = float(round(confidence * 100, 2))

        # threshold 65%
        if confidence_percent < 65:
            status = "⚠️ uncertain"
        else:
            status = "✅ confident"

        all_confidences.append(confidence_percent)

        results.append({
            "image_index": i+1,
            "prediction": result,
            "confidence": confidence_percent,
            "healthy_prob": float(round(healthy_prob * 100, 2)),
            "diseases_prob": float(round(diseases_prob * 100, 2)),
            "status": status
        })

    # 🔥 Final decision
    if diseases_count > len(results) / 2:
        final_decision = "diseases"
    else:
        final_decision = "healthy"

    avg_conf = float(round(sum(all_confidences) / len(all_confidences), 2))

    return jsonify({
        "total_images": len(results),
        "final_decision": final_decision,
        "average_confidence": avg_conf,
        "results": results
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)