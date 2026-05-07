



from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# =========================
# Load Model
# =========================
model = tf.keras.models.load_model(
    "saved_models/final_model_5classes2.h5"
)

# IMPORTANT:
# لازم نفس ترتيب الـ classes وقت التدريب
class_names = [
    "botrytis_gray_mold",
    "downy_mildew",
    "healthy",
    "pythium_root_rot",
    "tip_burn"
]

# =========================
# Config
# =========================
MAX_IMAGES = 10
IMAGE_SIZE = (224, 224)

# confidence threshold
CONFIDENCE_THRESHOLD = 65


# =========================
# Treatments Database
# =========================
treatments = {

    "botrytis_gray_mold": {
        "description": "Gray fungal mold affecting leaves and stems.",
        "treatment": "Remove infected leaves and apply fungicide."
    },

    "downy_mildew": {
        "description": "Yellow spots caused by fungal infection.",
        "treatment": "Reduce humidity and use proper fungicide."
    },

    "pythium_root_rot": {
        "description": "Root disease caused by overwatering.",
        "treatment": "Improve drainage and reduce watering."
    },

    "tip_burn": {
        "description": "Brown leaf edges due to nutrient imbalance.",
        "treatment": "Adjust calcium levels and watering schedule."
    }
}


# =========================
# Image Preprocessing
# =========================
def preprocess_image(img):

    img = img.resize(IMAGE_SIZE)

    img_array = np.array(
        img,
        dtype=np.float32
    )

    return img_array


# =========================
# Predict Route
# =========================
print("NEW API WORKING")
@app.route("/predict", methods=["POST"])
def predict():

    # check images
    if "images" not in request.files:
        return jsonify({
            "error": "No images uploaded"
        }), 400

    files = request.files.getlist("images")

    # max images validation
    if len(files) > MAX_IMAGES:
        return jsonify({
            "error": "Maximum 10 images allowed"
        }), 400

    # empty validation
    if len(files) == 0:
        return jsonify({
            "error": "Empty request"
        }), 400

    # =========================
    # Read Images
    # =========================
    images = []

    for file in files:

        img = Image.open(file).convert("RGB")

        processed = preprocess_image(img)

        images.append(processed)

    images = np.array(images)

    # =========================
    # Prediction
    # =========================
    predictions = model.predict(images, verbose=0)

    results = []

    diseases_count = 0

    all_confidences = []

    # =========================
    # Process Predictions
    # =========================
    for i, pred in enumerate(predictions):

        pred_index = np.argmax(pred)

        label = class_names[pred_index]

        confidence = float(
            pred[pred_index] * 100
        )

        all_confidences.append(confidence)

        # =====================
        # Status
        # =====================
        if confidence >= CONFIDENCE_THRESHOLD:
            status = "confident"
        else:
            status = "uncertain"

        # =====================
        # Healthy Case
        # =====================
        if label == "healthy":

            result = {
                "image_index": i + 1,

                "prediction": "healthy",

                "disease_name": None,

                "confidence": round(confidence, 2),

                "status": status,

                "message": "Plant looks healthy 🌱"
            }

        # =====================
        # Disease Case
        # =====================
        else:

            diseases_count += 1

            treatment_info = treatments.get(
                label,
                {}
            )

            result = {
                "image_index": i + 1,

                "prediction": "diseases",

                "disease_name": label,

                "confidence": round(confidence, 2),

                "status": status,

                "description": treatment_info.get(
                    "description",
                    ""
                ),

                "treatment": treatment_info.get(
                    "treatment",
                    ""
                )
            }

        results.append(result)

    # =========================
    # Final Decision
    # =========================
    if diseases_count > len(results) / 2:
        final_decision = "diseases"
    else:
        final_decision = "healthy"

    # average confidence
    avg_confidence = round(
        sum(all_confidences) / len(all_confidences),
        2
    )

    # =========================
    # Final Response
    # =========================
    return jsonify({

        "total_images": len(results),

        "final_decision": final_decision,

        "average_confidence": avg_confidence,

        "results": results

    })


# =========================
# Run Server
# =========================
if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )








# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port)