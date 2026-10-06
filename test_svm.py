import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("dataset/deceptive.csv")
df['label'] = df['deceptive'].map({'truthful': 0, 'deceptive': 1})
df = df.dropna(subset=['text', 'label'])
df['clean_text'] = df['text'].astype(str).str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))

X = df['clean_text'].tolist()
y = df['label'].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vec = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

svm = LinearSVC(random_state=42)
svm.fit(X_train_vec, y_train)
preds = svm.predict(X_test_vec)
print("SVM + Bigrams Accuracy:", accuracy_score(y_test, preds))
