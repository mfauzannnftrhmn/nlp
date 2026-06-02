"""
Flask API Backend for Sentiment Analysis Web App
"""

from flask import Flask, request, jsonify, send_from_directory
import pickle
import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')

# ─── Load Model ──────────────────────────────────────────────────────────────
def load_model():
    model_path = os.path.join(MODEL_DIR, "sentiment_model.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")

    if not all(os.path.exists(p) for p in [model_path, vectorizer_path, metrics_path]):
        return None, None, None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    return model, vectorizer, metrics

model, vectorizer, metrics = load_model()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "status": "ready" if model is not None else "model_not_found",
        "message": "Model loaded successfully" if model else "Run train_model.py first"
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    if metrics is None:
        return jsonify({"error": "Model not trained yet. Run train_model.py first."}), 404
    return jsonify(metrics)

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not trained yet. Run train_model.py first."}), 404

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400

    cleaned = preprocess_text(text)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]

    neg_prob = round(float(probabilities[0]) * 100, 2)
    pos_prob = round(float(probabilities[1]) * 100, 2)
    confidence = pos_prob if prediction == 1 else neg_prob

    return jsonify({
        "text": text,
        "sentiment": "positive" if prediction == 1 else "negative",
        "label": int(prediction),
        "confidence": confidence,
        "probabilities": {
            "negative": neg_prob,
            "positive": pos_prob
        }
    })

if __name__ == '__main__':
    print("=" * 50)
    print("  Sentiment Analysis API - Flask Server")
    print("=" * 50)
    if model:
        print(f"  Model loaded: Multinomial Naive Bayes")
        print(f"  Accuracy: {metrics['accuracy']}%")
    else:
        print("  WARNING: Model not found. Run train_model.py first!")
    print(f"  Server: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
