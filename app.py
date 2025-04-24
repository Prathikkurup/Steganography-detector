from flask import Flask, request, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('stego_classifier.h5')

# Ensure that the 'uploads' folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize to match model input size
    img = img / 255.0  # Normalize
    img = np.reshape(img, (1, 128, 128, 3))  # Reshape for the model
    return img

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Preprocess and predict
    img = preprocess_image(file_path)
    prediction = model.predict(img)

    # Determine result
    if prediction < 0.5:
        result = 'Clean Image'
    else:
        result = 'Stego Image'

    return render_template('result.html', result=result)

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
