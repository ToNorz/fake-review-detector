# False Review Analyzer

This project is a Machine Learning-based fake review detection system, built as a college project. It classifies hotel reviews into two categories: **Truthful** (Real) and **Deceptive** (Fake).

## Methodology

The project follows the concepts discussed in the reference paper: *"Research on false review detection Methods: A state-of-the-art review"*.

Specifically, it uses **Review Dimension Features** by converting the review text into TF-IDF (Term Frequency-Inverse Document Frequency) numerical vectors (using both Unigrams and Bigrams). These vectors are then passed into a **Support Vector Machine (SVM)** to distinguish between genuine and deceptive opinion spam. This pipeline achieves roughly **89% accuracy**.

## Dataset
The model is trained on the **Deceptive Opinion Spam Corpus** (Ott et al.), which contains 1,600 hotel reviews (800 truthful, 800 deceptive). The dataset is located at `dataset/deceptive.csv`.

## Project Structure
- `false_review_analyzer.py`: Main script to load the dataset, train the SVM model, and evaluate its accuracy.
- `predict_review.py`: A script demonstrating how to load the saved model and vectorizer to predict whether a custom text review is fake or real in the terminal.
- `app.py`: A **Web UI** built with Streamlit to present the model interactively.
- `requirements.txt`: Python dependencies needed to run the project.

## Installation

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train the model**:
   Run the training script to build the model from the dataset and generate the `.pkl` files and confusion matrix plot.
   ```bash
   python false_review_analyzer.py
   ```
2. **Present the Web UI (Recommended)**:
   Run the Streamlit app to show off an interactive web interface for your presentation.
   ```bash
   streamlit run app.py
   ```
3. **Test in Terminal**:
   Run the prediction script to see the model classify custom reviews in the command line.
   ```bash
   python predict_review.py
   ```
