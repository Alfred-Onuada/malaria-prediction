from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('./ML/model.h5')

# TODO: add cors so request can only be made from the node server

# Function to preprocess the image
# Function to preprocess the image
def preprocess_image(file_storage, target_size=(128, 128)):
    # Convert file storage to numpy array
    nparr = np.fromstring(file_storage.read(), np.uint8)
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, target_size)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

@app.route("/predict", methods=["POST"])
def predict():
    # Check if the request contains an image file
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})
    
    file = request.files["image"]
    
    # Check if the file is empty
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    
    # Check if the file is an image
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Preprocess the image
        image = preprocess_image(file)
        
        # Make prediction
        prediction = model.predict(image)
        
        # Return the prediction result
        return jsonify({"prediction": float(prediction[0][0])})
    
    else:
        return jsonify({"error": "Invalid file format"})

if __name__ == '__main__':
    PORT = os.getenv('PORT') if os.getenv('PORT') else 3025
    app.run(debug=True, port=PORT, host="0.0.0.0")
