



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

    "pythium_root_rot": {
        "description": "Root disease caused by Pythium fungus in hydroponic systems.",
        "treatment": [
            "Increase water aeration to maintain dissolved oxygen",
            "Keep water temperature between 18–20°C",
            "Add Trichoderma to suppress Pythium naturally",
            "Apply Mefenoxam or Fosetyl-Al to the nutrient solution"
        ]
    },

    "botrytis_gray_mold": {
        "description": "Gray fungal mold affecting leaves and stems.",
        "treatment": [
            "Use fans to reduce humidity around plants",
            "Remove infected leaves immediately before spores spread",
            "Spray Bacillus subtilis as a biological fungicide",
            "Apply Iprodione or Fenhexamid fungicides"
        ]
    },

    "downy_mildew": {
        "description": "Fungal disease causing yellow spots and mildew.",
        "treatment": [
            "Keep humidity below 85% with good ventilation",
            "Spray 1 tsp sodium bicarbonate per liter of water",
            "Apply Mandipropamid or Dimethomorph fungicides",
            "Use Bremia-resistant lettuce varieties"
        ]
    },

    "tip_burn": {
        "description": "Leaf edge burn caused by calcium deficiency.",
        "treatment": [
            "Run fans to circulate air around inner leaves",
            "Foliar spray 0.5% calcium chloride on leaves",
            "Lower nutrient solution EC to improve calcium absorption",
            "Reduce high-intensity light hours to slow excessive growth"
        ]
    },

    "healthy": {
        "description": "Plant is healthy with no detected disease.",
        "treatment": [
            "Keep EC between 0.8–1.6 and pH between 5.5–6.5",
            "Maintain water temperature at 18–22°C",
            "Provide 14–16 hours of light daily",
            "Check roots weekly — healthy roots are white and firm"
        ]
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

        try:

            img = Image.open(file).convert("RGB")

            processed = preprocess_image(img)

            images.append(processed)

        except Exception:

            return jsonify({
                "error": f"Invalid image: {file.filename}"
            }), 400
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

            treatment_info = treatments.get(
                "healthy",
                {}
            )

            result = {
                "image_index": i + 1,

                "prediction": "healthy",



                "confidence": round(confidence, 2),

                "status": status,

                "message": "Plant looks healthy 🌱",

                "description": treatment_info.get(
                    "description",
                    ""
                ),

                "treatment": treatment_info.get(
                    "treatment",
                    []
                )
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
                    []
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