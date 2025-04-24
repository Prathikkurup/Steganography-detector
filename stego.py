
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

# Load images from folders (clean and stego)
def load_images_from_folders(folder):
    images = []
    labels = []
    
    # Paths for clean and stego images
    clean_folder = os.path.join(folder, 'clean')
    stego_folder = os.path.join(folder, 'stego')

    # Load clean images
    for filename in os.listdir(clean_folder):
        img = cv2.imread(os.path.join(clean_folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            labels.append(0)  # 0 for clean images

    # Load stego images
    for filename in os.listdir(stego_folder):
        img = cv2.imread(os.path.join(stego_folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            labels.append(1)  # 1 for stego images

    return images, labels

# Feature extraction - using image statistics (e.g., histogram of pixel values)
def extract_features(images):
    features = []
    for img in images:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist.flatten()  # Flatten the histogram into a 1D array
        features.append(hist)
    return np.array(features)

# Detect steganography in a user-provided image
def detect_steganography(user_image_path, classifier, scaler):
    img = cv2.imread(user_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {user_image_path}")
        return

    # Preprocess the image: extract features (histogram)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.flatten()

    # Scale the features using the same scaler used during training
    hist_scaled = scaler.transform([hist])

    # Make a prediction
    prediction = classifier.predict(hist_scaled)
    
    # Output the result
    if prediction[0] == 1:
        print(f"The image '{user_image_path}' contains steganography (stego).")
    else:
        print(f"The image '{user_image_path}' is clean (no steganography).")

# Path to your dataset
image_path = r"D:\ML\train"

# Load datasets from the specified path
images, labels = load_images_from_folders(image_path)

# Check if images were loaded correctly
print(f"Loaded {len(images)} images.")

# Extract features
features = extract_features(images)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an SVM classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Predict on test data
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))



# User input: provide the path to the image you want to detect
user_image_path = r"D:\ML\test\stego\image_04001_html_0.png"  # Replace with your own path

# Detect steganography in the user-provided image
detect_steganography(user_image_path, classifier, scaler)