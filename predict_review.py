import pickle
import re

MODEL_PATH = "svm_review_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

def clean_text(text):
    """Basic text cleaning for NLP. Must match the training preprocessing."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_artifacts():
    """Loads the trained model and vectorizer."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        print("Error: Model or vectorizer not found. Please run false_review_analyzer.py first.")
        return None, None

def predict_review(model, vectorizer, text):
    """Cleans text, transforms it, and predicts."""
    # Preprocess the text
    cleaned = clean_text(text)
    
    # Transform using vectorizer
    text_features = vectorizer.transform([cleaned])

    # Predict
    prediction = model.predict(text_features)[0]
    confidence = model.predict_proba(text_features)[0]

    # Map label
    label = "Deceptive (Fake)" if prediction == 1 else "Truthful (Real)"
    confidence_score = confidence[prediction] * 100

    return label, confidence_score

def main():
    model, vectorizer = load_artifacts()
    if model is None or vectorizer is None:
        return

    sample_reviews = [
        "I had a terrible stay at this hotel. The room was dirty, the staff was extremely rude, and the food made me sick.",
        "The Hilton Chicago was absolutely breathtaking! From the moment we walked in, the concierge treated us like royalty. Will definitely return!"
    ]
    
    print("Testing the loaded model with sample reviews...")
    for review in sample_reviews:
        label, conf = predict_review(model, vectorizer, review)
        print(f"\nReview: '{review}'")
        print(f"Prediction: {label}")
        print(f"Confidence: {conf:.2f}%")

if __name__ == "__main__":
    main()
