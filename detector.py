import os
import sys
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Directories for the dataset
REAL_DIR = "dataset/real"
FAKE_DIR = "dataset/fake"
MODEL_PATH = "rf_deepfake_model.pkl"

def extract_features(img_path, img_size=(64, 64)):
    """Extract basic features for a minimal deepfake detector."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Resize image to a standard small size to limit feature dimensions
    img_resized = cv2.resize(img, img_size)

    # 1. Spatial features: Flattened grayscale image
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    spatial_features = gray.flatten()

    # 2. Texture anomalies: Variance of Laplacian (blur detection)
    # Deepfakes sometimes exhibit blending artifacts or blurriness
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Combine features into a single 1D array
    features = np.append(spatial_features, laplacian_var)
    return features

def load_dataset():
    """Loads images from the real and fake directories."""
    X, y = [], []

    print(f"Loading real images from {REAL_DIR}...")
    for file in os.listdir(REAL_DIR):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            feats = extract_features(os.path.join(REAL_DIR, file))
            if feats is not None:
                X.append(feats)
                y.append(0) # 0 = Real

    print(f"Loading fake images from {FAKE_DIR}...")
    for file in os.listdir(FAKE_DIR):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            feats = extract_features(os.path.join(FAKE_DIR, file))
            if feats is not None:
                X.append(feats)
                y.append(1) # 1 = Fake

    return np.array(X), np.array(y)

def main():
    # 1. Setup dataset directories
    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(FAKE_DIR, exist_ok=True)

    # Ensure there are images in the directories
    if not os.listdir(REAL_DIR) or not os.listdir(FAKE_DIR):
        print(f"Dataset directories are empty or missing!")
        print(f"Please place real face images in '{REAL_DIR}' and deepfake images in '{FAKE_DIR}'")
        print("Then run this script again.")
        sys.exit(0)

    # 2. Load and extract features
    X, y = load_dataset()
    if len(X) == 0:
        print("No valid images found. Exiting.")
        sys.exit(1)

    print(f"\nTotal samples loaded: {len(X)} (Real: {np.sum(y==0)}, Fake: {np.sum(y==1)})")

    # 3. Train-Test Split (80% training, 20% testing)
    if len(X) < 5:
        print("Please provide at least a few images in both folders to train properly.")
        sys.exit(1)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Initialize and Train Random Forest
    print("\nTraining Random Forest model...")
    # n_estimators=100 creates 100 decision trees
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)

    # 5. Evaluate the model
    print("Evaluating model...")
    y_pred = rf_classifier.predict(X_test)
    
    print("\n--- Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    try:
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    except ValueError:
        print(classification_report(y_test, y_pred))

    # 6. Save the trained model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(rf_classifier, f)
    print(f"\nModel saved successfully to {MODEL_PATH}")

if __name__ == "__main__":
    main()
