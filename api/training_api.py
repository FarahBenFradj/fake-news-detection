from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Global variables
DATA_FILE = 'training_data.csv'
MODEL_FILE = 'fake_news_model.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'

def preprocess_text(text):
    """Clean and preprocess text data"""
    if not text or pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

@app.route('/')
def home():
    return jsonify({
        "message": "Training API - Data Collection & Model Training",
        "endpoints": {
            "/add_data": "POST - Add training data",
            "/batch_load": "POST - Load CSV file with multiple samples",
            "/get_data": "GET - View all training data",
            "/train": "POST - Train the model with 5-fold cross-validation",
            "/status": "GET - Get training status"
        }
    })

@app.route('/add_data', methods=['POST'])
def add_data():
    """Add single training sample (text + label)"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data or 'label' not in data:
            return jsonify({
                "error": "Please provide 'text' and 'label' (0=real, 1=fake)"
            }), 400
        
        text = data['text']
        label = int(data['label'])
        
        if label not in [0, 1]:
            return jsonify({"error": "Label must be 0 (real) or 1 (fake)"}), 400
        
        if len(text.strip()) < 10:
            return jsonify({"error": "Text too short (minimum 10 characters)"}), 400
        
        # Load existing data or create new
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
        else:
            df = pd.DataFrame(columns=['text', 'label'])
        
        # Add new data
        new_row = pd.DataFrame({
            'text': [text],
            'label': [label]
        })
        
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(DATA_FILE, index=False, encoding='utf-8')
        
        return jsonify({
            "message": "Data added successfully",
            "total_samples": len(df),
            "real_count": len(df[df['label'] == 0]),
            "fake_count": len(df[df['label'] == 1])
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch_load', methods=['POST'])
def batch_load():
    """Load entire CSV file with training data (faster than /add_data)"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                "error": "No file provided. Send CSV with columns: 'text' and 'label'"
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be CSV format"}), 400
        
        # Load CSV
        df_new = pd.read_csv(file)
        
        # Validate columns
        if 'text' not in df_new.columns or 'label' not in df_new.columns:
            return jsonify({
                "error": "CSV must contain 'text' and 'label' columns"
            }), 400
        
        # Validate labels
        if not all(df_new['label'].isin([0, 1])):
            return jsonify({
                "error": "Labels must be 0 (real) or 1 (fake)"
            }), 400
        
        # Load existing data if it exists
        if os.path.exists(DATA_FILE):
            df_existing = pd.read_csv(DATA_FILE)
            df_new = pd.concat([df_existing, df_new], ignore_index=True)
        
        # Remove duplicates
        df_new = df_new.drop_duplicates(subset=['text'], keep='first')
        
        # Save
        df_new.to_csv(DATA_FILE, index=False, encoding='utf-8')
        
        return jsonify({
            "message": f"Batch loaded {len(df_new)} samples successfully",
            "total_samples": len(df_new),
            "real_count": len(df_new[df_new['label'] == 0]),
            "fake_count": len(df_new[df_new['label'] == 1])
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_data', methods=['GET'])
def get_data():
    """Get all training data"""
    try:
        if not os.path.exists(DATA_FILE):
            return jsonify({
                "message": "No training data yet",
                "total_samples": 0
            })
        
        df = pd.read_csv(DATA_FILE)
        
        return jsonify({
            "total_samples": len(df),
            "real_count": int(len(df[df['label'] == 0])),
            "fake_count": int(len(df[df['label'] == 1])),
            "samples": len(df)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Train model with 5-fold cross-validation"""
    try:
        if not os.path.exists(DATA_FILE):
            return jsonify({"error": "No training data available"}), 400
        
        print("\n" + "="*70)
        print("STARTING MODEL TRAINING WITH 5-FOLD CROSS-VALIDATION")
        print("="*70)
        
        # Load data
        df = pd.read_csv(DATA_FILE)
        
        if len(df) < 20:
            return jsonify({
                "error": f"Need at least 20 samples to train. Current: {len(df)}"
            }), 400
        
        print(f"\nLoaded {len(df)} samples for training")
        print(f"Real news: {len(df[df['label'] == 0])}")
        print(f"Fake news: {len(df[df['label'] == 1])}")
        
        # Preprocess
        print("\nPreprocessing text...")
        df['cleaned_text'] = df['text'].apply(preprocess_text)
        df = df[df['cleaned_text'].str.len() > 0]
        
        X = df['cleaned_text']
        y = df['label']
        
        # Split data for final test evaluation (not used in CV)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nData split:")
        print(f"Training set: {len(X_train)} samples (used for CV)")
        print(f"Test set: {len(X_test)} samples (final evaluation)")
        
        # TF-IDF Vectorization
        print("\nVectorizing text...")
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
        # Train base model
        print("\nTraining Logistic Regression model...")
        model = LogisticRegression(max_iter=1000, random_state=42)
        
        # Cross-validation (5-fold on training set)
        print("\n" + "-"*70)
        print("CROSS-VALIDATION RESULTS (5-Fold)")
        print("-"*70)
        cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
        
        print("\nIndividual fold scores:")
        for i, score in enumerate(cv_scores, 1):
            print(f"  Fold {i}: {score:.4f}")
        
        print(f"\nCross-Validation Summary:")
        print(f"  Mean Accuracy: {cv_scores.mean():.4f}")
        print(f"  Std Deviation: {cv_scores.std():.4f}")
        print(f"  95% Confidence Interval: {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")
        
        # Train final model on full training set
        model.fit(X_train_tfidf, y_train)
        
        # Test set evaluation
        y_pred = model.predict(X_test_tfidf)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n" + "-"*70)
        print("FINAL TEST SET EVALUATION (20% held-out data)")
        print("-"*70)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"{'':15} Predicted Real  Predicted Fake")
        print(f"Actually Real:  {cm[0][0]:>14} {cm[0][1]:>14}")
        print(f"Actually Fake:  {cm[1][0]:>14} {cm[1][1]:>14}")
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nDetailed Metrics:")
        print(f"  Precision (Fake): {precision:.4f}")
        print(f"  Recall (Fake): {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        # Save model and vectorizer
        print("\nSaving model and vectorizer...")
        joblib.dump(model, MODEL_FILE)
        joblib.dump(tfidf, VECTORIZER_FILE)
        
        # Save training info
        training_info = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": len(df),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "test_accuracy": float(test_accuracy),
            "confusion_matrix": cm.tolist(),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        
        with open('training_info.json', 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print("\n" + "="*70)
        print("✓ TRAINING COMPLETE!")
        print("="*70)
        
        return jsonify({
            "message": "Model trained successfully!",
            "cv_mean_accuracy": f"{cv_scores.mean():.4f}",
            "cv_std": f"{cv_scores.std():.4f}",
            "cv_scores": [round(s, 4) for s in cv_scores.tolist()],
            "test_accuracy": f"{test_accuracy:.4f}",
            "precision": f"{precision:.4f}",
            "recall": f"{recall:.4f}",
            "f1_score": f"{f1:.4f}",
            "confusion_matrix": cm.tolist(),
            "total_samples": len(df),
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        })
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Get training status and model info"""
    try:
        response = {
            "data_available": os.path.exists(DATA_FILE),
            "model_trained": os.path.exists(MODEL_FILE),
            "vectorizer_available": os.path.exists(VECTORIZER_FILE)
        }
        
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            response["total_samples"] = len(df)
            response["real_count"] = int(len(df[df['label'] == 0]))
            response["fake_count"] = int(len(df[df['label'] == 1]))
        
        if os.path.exists('training_info.json'):
            with open('training_info.json', 'r') as f:
                response["last_training"] = json.load(f)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("TRAINING API - Data Collection & Model Training")
    print("="*70)
    print("\nAPI will be available at: http://localhost:5001")
    print("\nEndpoints:")
    print("  POST /add_data       - Add single training sample")
    print("  POST /batch_load     - Load entire CSV file")
    print("  GET  /get_data       - View data statistics")
    print("  POST /train          - Train with 5-fold CV")
    print("  GET  /status         - Get status")
    print("\nUsage:")
    print("  1. Run prepare_data.py to split datasets")
    print("  2. Use /batch_load to upload training_data.csv")
    print("  3. Use /train to train with cross-validation")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)