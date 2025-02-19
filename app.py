from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import os
from PIL import Image

app = Flask(__name__) 

# Create an upload folder if not exists
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load YOLOv5 model
MODEL_PATH = "yolov5s.pt"  # Ensure you have the model file
model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Load and process image with YOLOv5
    img = Image.open(file_path)
    results = model(img)

    # Extract detected objects
    detected_objects = []
    for obj in results.xyxy[0]:  # Using xyxy format for bounding boxes
        class_id = int(obj[5])
        label = model.names[class_id]
        detected_objects.append(label)

    return jsonify({
        "objects": detected_objects,
        "image_url": f"/uploads/{file.filename}"
    })

@app.route("/uploads/<filename>")
def get_uploaded_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Port for hosting
    app.run(debug=True, host="0.0.0.0", port=port)
