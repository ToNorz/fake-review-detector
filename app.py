import streamlit as st
import pickle
import re

# Load the model and vectorizer
@st.cache_resource
def load_artifacts():
    try:
        with open("svm_review_model.pkl", 'rb') as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer.pkl", 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.title("🏨 False Review Analyzer")
st.write("Welcome to the Fake Hotel Review Detection system. Enter a review below to analyze its authenticity.")

model, vectorizer = load_artifacts()

if model is None:
    st.error("Model files not found. Please run `python false_review_analyzer.py` first to train the model.")
else:
    user_input = st.text_area("Review Text", height=150, placeholder="Type a hotel review here...")
    
    if st.button("Analyze Review"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            # Preprocess and predict
            cleaned = clean_text(user_input)
            features = vectorizer.transform([cleaned])
            
            prediction = model.predict(features)[0]
            confidence = model.predict_proba(features)[0]
            
            label = "Deceptive (Fake)" if prediction == 1 else "Truthful (Real)"
            conf_score = confidence[prediction] * 100
            
            st.markdown("### Results")
            if prediction == 1:
                st.error(f"**Prediction:** {label}")
            else:
                st.success(f"**Prediction:** {label}")
                
            st.info(f"**Confidence Score:** {conf_score:.2f}%")
