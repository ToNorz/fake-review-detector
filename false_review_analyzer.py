"""
False Review Analyzer
---------------------
This script trains a Machine Learning model to detect deceptive (fake) opinion spam 
in hotel reviews using TF-IDF features and a Random Forest Classifier.
Reference: "Research on false review detection Methods: A state-of-the-art review"
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATASET_PATH = "dataset/deceptive.csv"
MODEL_PATH = "rf_review_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: {DATASET_PATH} not found.")
        return

    if 'text' not in df.columns or 'deceptive' not in df.columns:
        print("Error: Dataset must contain 'text' and 'deceptive' columns.")
        return

    print(f"Dataset loaded. Total reviews: {len(df)}")
    
    # Map 'truthful' -> 0, 'deceptive' -> 1
    df['label'] = df['deceptive'].map({'truthful': 0, 'deceptive': 1})
    
    df = df.dropna(subset=['text', 'label'])
    
    # Convert to python list to avoid pyarrow issues with sklearn train_test_split
    X = df['text'].astype(str).tolist()
    y = df['label'].astype(int).tolist()

    print(f"Class distribution: Real: {y.count(0)}, Fake: {y.count(1)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Random Forest model...")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train_vec, y_train)

    print("Evaluating model...")
    y_pred = rf_classifier.predict(X_test_vec)
    
    print("\n--- Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real (0)', 'Fake (1)']))

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(rf_classifier, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print(f"\nModel saved successfully to {MODEL_PATH}")
    print(f"Vectorizer saved successfully to {VECTORIZER_PATH}")

if __name__ == "__main__":
    main()
