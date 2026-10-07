"""
False Review Analyzer
---------------------
This script trains a Machine Learning model to detect deceptive (fake) opinion spam 
in hotel reviews using TF-IDF features and a Support Vector Machine (SVM).
Reference: "Research on false review detection Methods: A state-of-the-art review"
"""
import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

DATASET_PATH = "dataset/deceptive.csv"
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

def load_dataset(filepath):
    print(f"Loading dataset from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(df):
    if 'text' not in df.columns or 'deceptive' not in df.columns:
        print("Error: Dataset must contain 'text' and 'deceptive' columns.")
        return None, None

    print(f"Dataset loaded. Total raw reviews: {len(df)}")
    df['label'] = df['deceptive'].map({'truthful': 0, 'deceptive': 1})
    df = df.dropna(subset=['text', 'label'])
    
    print("Cleaning text data...")
    df['clean_text'] = df['text'].apply(clean_text)

    X = df['clean_text'].tolist()
    y = df['label'].astype(int).tolist()
    print(f"Class distribution: Real: {y.count(0)}, Fake: {y.count(1)}")
    return X, y

def extract_features(X_train, X_test, max_features=10000):
    print("Extracting TF-IDF features (Unigrams & Bigrams)...")
    # Adding bigrams (2-word combinations) dramatically improves accuracy on text
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer

def train_model(X_train_vec, y_train):
    print("Training Support Vector Machine (SVM) model...")
    # SVMs generally outperform Random Forests on text classification
    svm = LinearSVC(random_state=42)
    # CalibratedClassifierCV allows the SVM to output probability percentages
    model = CalibratedClassifierCV(svm)
    model.fit(X_train_vec, y_train)
    return model

def evaluate_model(model, X_test_vec, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred) * 100
    print("\n--- Results ---")
    print(f"Accuracy: {acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real (0)', 'Fake (1)']))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix (Accuracy: {acc:.1f}%)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\nSaved confusion matrix plot to 'confusion_matrix.png'")

def save_artifacts(model, vectorizer):
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print(f"Model saved successfully to {MODEL_PATH}")
    print(f"Vectorizer saved successfully to {VECTORIZER_PATH}")

def main():
    df = load_dataset(DATASET_PATH)
    if df is None: return
    X, y = preprocess_data(df)
    if X is None: return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_vec, X_test_vec, vectorizer = extract_features(X_train, X_test)
    model = train_model(X_train_vec, y_train)
    evaluate_model(model, X_test_vec, y_test)
    save_artifacts(model, vectorizer)

if __name__ == "__main__":
    main()
