import cv2
import numpy as np
import os
import joblib

# Load the pre-trained SVM classifier and scaler
classifier = joblib.load('models/svm_classifier.pkl')
scaler = joblib.load('models/scaler.pkl')

# Feature extraction for a single image (using histogram of pixel values)
def extract_features(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist.flatten()

# Function to detect steganography in a single image
def detect_steganography(image_path):
    # Load the image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Extract features (histogram) and scale them
    features = extract_features(img)
    features_scaled = scaler.transform([features])

    # Make a prediction
    prediction = classifier.predict(features_scaled)
    
    # Output the result
    if prediction[0] == 1:
        print(f"The image '{image_path}' contains steganography (stego).")
    else:
        print(f"The image '{image_path}' is clean (no steganography).")

# Detect steganography in a folder of images (optional for batch processing)
def detect_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        detect_steganography(image_path)

if __name__ == "__main__":
    # Example usage for detecting in a single image
    user_image_path = r"D:\ML\test\stego\image_04001_html_0.png"  # Replace with your image path
    detect_steganography(user_image_path)

    # Optional: Detect steganography in all images in a folder
    # folder_path = r"D:\ML\test\stego"  # Replace with your folder path
    # detect_in_folder(folder_path)
