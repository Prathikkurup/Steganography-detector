import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import os
import joblib
from tqdm import tqdm  # Import tqdm for progress bar

# Load images from folders (clean and stego)
def load_images_from_folders(folder):
    images = []
    labels = []
    
    # Paths for clean and stego images
    clean_folder = os.path.join(folder, 'clean')
    stego_folder = os.path.join(folder, 'stego')

    # Load clean images
    print("Loading clean images...")
    for filename in tqdm(os.listdir(clean_folder)):
        img = cv2.imread(os.path.join(clean_folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            labels.append(0)  # 0 for clean images

    # Load stego images
    print("Loading stego images...")
    for filename in tqdm(os.listdir(stego_folder)):
        img = cv2.imread(os.path.join(stego_folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            labels.append(1)  # 1 for stego images

    print(f"Loaded {len(images)} images.")
    return images, labels

# Feature extraction - using image statistics (e.g., histogram of pixel values)
def extract_features(images):
    features = []
    print("Extracting features...")
    for idx, img in enumerate(tqdm(images)):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist.flatten()  # Flatten the histogram into a 1D array
        features.append(hist)

        # Print progress every 100 images
        if idx % 100 == 0:
            print(f"Processed {idx} images for feature extraction...")

    return np.array(features)

# Main function to train the model
def train_model():
    # Path to your dataset
    image_path = r"D:\ML\train"

    # Load datasets from the specified path
    images, labels = load_images_from_folders(image_path)

    # Extract features
    features = extract_features(images)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train an SVM classifier
    print("Training the SVM classifier...")
    classifier = SVC(kernel='linear')
    classifier.fit(X_train, y_train)

    # Predict on test data
    print("Making predictions on the test set...")
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model and scaler
    print("Saving the model and scaler...")
    joblib.dump(classifier, 'models/svm_classifier.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    train_model()

