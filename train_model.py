"""
Sentiment Analysis Model Training Script
Dataset: Amazon Product Reviews (amazon_cells_labelled.txt)
Algorithm: Multinomial Naive Bayes
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
import re

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "amazon_cells_labelled.txt")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# ─── Preprocessing ───────────────────────────────────────────────────────────
def preprocess_text(text):
    """Clean and normalize text."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─── Load Dataset ────────────────────────────────────────────────────────────
print("=" * 60)
print("   SENTIMENT ANALYSIS - MODEL TRAINING")
print("   Algorithm: Multinomial Naive Bayes")
print("   Dataset:   Amazon Product Reviews")
print("=" * 60)

print("\n[1/5] Loading dataset...")
df = pd.read_csv(DATASET_PATH, sep='\t', header=None, names=['text', 'label'])
df = df.dropna()
df['text'] = df['text'].astype(str)
df['label'] = df['label'].astype(int)

print(f"      Total samples     : {len(df)}")
print(f"      Positive (1)      : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
print(f"      Negative (0)      : {(df['label'] == 0).sum()} ({(df['label']==0).mean()*100:.1f}%)")

# ─── Preprocess ──────────────────────────────────────────────────────────────
print("\n[2/5] Preprocessing text...")
df['cleaned'] = df['text'].apply(preprocess_text)

# ─── Split ───────────────────────────────────────────────────────────────────
print("\n[3/5] Splitting dataset (80% train / 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)
print(f"      Training samples  : {len(X_train)}")
print(f"      Testing samples   : {len(X_test)}")

# ─── Vectorize ───────────────────────────────────────────────────────────────
print("\n[4/5] Vectorizing text (TF-IDF, max 5000 features)...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english',
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ─── Train ───────────────────────────────────────────────────────────────────
print("\n[5/5] Training Multinomial Naive Bayes model...")
model = MultinomialNB(alpha=0.5)
model.fit(X_train_vec, y_train)

# ─── Evaluate ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("   EVALUATION RESULTS")
print("=" * 60)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], output_dict=True)

print(f"\n  Accuracy  : {accuracy*100:.2f}%")
print(f"\n  Confusion Matrix:")
print(f"              Predicted")
print(f"              Neg    Pos")
print(f"  Actual Neg  {cm[0][0]:<6} {cm[0][1]}")
print(f"         Pos  {cm[1][0]:<6} {cm[1][1]}")

print(f"\n  Classification Report:")
print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print(f"  {'-'*44}")
for cls in ['Negative', 'Positive']:
    r = report[cls]
    print(f"  {cls:<12} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f}")

# ─── Save Model ──────────────────────────────────────────────────────────────
metrics = {
    "algorithm": "Multinomial Naive Bayes",
    "dataset": "Amazon Product Reviews",
    "total_samples": len(df),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "accuracy": round(accuracy * 100, 2),
    "confusion_matrix": cm.tolist(),
    "precision_negative": round(report['Negative']['precision'] * 100, 2),
    "recall_negative": round(report['Negative']['recall'] * 100, 2),
    "f1_negative": round(report['Negative']['f1-score'] * 100, 2),
    "precision_positive": round(report['Positive']['precision'] * 100, 2),
    "recall_positive": round(report['Positive']['recall'] * 100, 2),
    "f1_positive": round(report['Positive']['f1-score'] * 100, 2),
    "support_negative": int(report['Negative']['support']),
    "support_positive": int(report['Positive']['support']),
}

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

with open(VECTORIZER_PATH, 'wb') as f:
    pickle.dump(vectorizer, f)

with open(METRICS_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 60)
print("   MODEL SAVED SUCCESSFULLY!")
print("=" * 60)
print(f"\n  Model      : {MODEL_PATH}")
print(f"  Vectorizer : {VECTORIZER_PATH}")
print(f"  Metrics    : {METRICS_PATH}")
print("\n  Run `python app.py` to start the web server.")
print("=" * 60)
