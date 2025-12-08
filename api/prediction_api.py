from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Load model and vectorizer
print("Loading model and vectorizer...")
try:
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"⚠ Warning: Could not load model - {e}")
    model = None
    vectorizer = None

def preprocess_text(text):
    """Clean and preprocess text data"""
    if not text or text.strip() == "":
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

def get_top_features(text_vectorized, prediction, n=10):
    """Get top features that influenced the prediction"""
    try:
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get coefficients for the prediction class
        if prediction == 1:  # Fake
            coef = model.coef_[0]
        else:  # Real
            coef = -model.coef_[0]
        
        # Get non-zero features in the text
        text_features = text_vectorized.toarray()[0]
        non_zero_indices = np.where(text_features > 0)[0]
        
        # Calculate importance scores
        importance_scores = []
        for idx in non_zero_indices:
            word = feature_names[idx]
            weight = coef[idx]
            tfidf_score = text_features[idx]
            importance = weight * tfidf_score
            importance_scores.append((word, importance, weight))
        
        # Sort by importance
        importance_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return importance_scores[:n]
    
    except Exception as e:
        print(f"Error in get_top_features: {e}")
        return []

def explain_prediction(text, prediction, probabilities, top_features):
    """Generate human-readable explanation"""
    
    if prediction == 1:  # FAKE
        reasons = []
        
        # Check for sensational words
        sensational_words = ['shocking', 'unbelievable', 'amazing', 'incredible', 
                             'won\'t believe', 'doctors hate', 'secret', 'trick']
        found_sensational = [w for w in sensational_words if w in text.lower()]
        
        if found_sensational:
            reasons.append(f"Contains sensational language: {', '.join(found_sensational[:3])}")
        
        # Check for clickbait patterns
        if '!' in text and text.count('!') > 2:
            reasons.append("Excessive use of exclamation marks (clickbait indicator)")
        
        if '?' in text and 'you' in text.lower():
            reasons.append("Uses questions directed at reader (common in fake news)")
        
        # Add top predictive words
        if top_features:
            fake_words = [word for word, _, weight in top_features if weight > 0][:3]
            if fake_words:
                reasons.append(f"Key fake indicators: {', '.join(fake_words)}")
        
        # Confidence-based explanation
        fake_conf = probabilities[1]
        if fake_conf > 0.9:
            confidence_level = "very high confidence"
        elif fake_conf > 0.75:
            confidence_level = "high confidence"
        else:
            confidence_level = "moderate confidence"
        
        explanation = {
            "verdict": "FAKE NEWS",
            "confidence_level": confidence_level,
            "reasons": reasons if reasons else ["Pattern matches fake news characteristics"],
            "top_fake_indicators": [word for word, _, weight in top_features if weight > 0][:5],
            "summary": f"This text is classified as fake news with {confidence_level}. The language patterns and word choices are typical of misinformation."
        }
    
    else:  # REAL
        reasons = []
        
        # Check for factual language
        factual_words = ['announced', 'reported', 'according', 'study', 'research', 
                        'said', 'government', 'official', 'university']
        found_factual = [w for w in factual_words if w in text.lower()]
        
        if found_factual:
            reasons.append(f"Uses factual language: {', '.join(found_factual[:3])}")
        
        # Check for formal structure
        if '.' in text and text.count('.') >= 2:
            reasons.append("Proper sentence structure and punctuation")
        
        # Add top predictive words
        if top_features:
            real_words = [word for word, _, weight in top_features if weight < 0][:3]
            if real_words:
                reasons.append(f"Key credibility indicators: {', '.join(real_words)}")
        
        # Confidence-based explanation
        real_conf = probabilities[0]
        if real_conf > 0.9:
            confidence_level = "very high confidence"
        elif real_conf > 0.75:
            confidence_level = "high confidence"
        else:
            confidence_level = "moderate confidence"
        
        explanation = {
            "verdict": "REAL NEWS",
            "confidence_level": confidence_level,
            "reasons": reasons if reasons else ["Pattern matches credible news characteristics"],
            "top_credibility_indicators": [word for word, _, weight in top_features if weight < 0][:5],
            "summary": f"This text is classified as real news with {confidence_level}. The language is factual and follows journalistic standards."
        }
    
    return explanation

@app.route('/')
def home():
    return jsonify({
        "message": "Prediction API - Fake News Detection with Explainability",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "/predict": "POST - Predict with explanation",
            "/health": "GET - Health check"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if news is fake or real with explanation"""
    try:
        if model is None or vectorizer is None:
            return jsonify({
                "error": "Model not loaded. Please train the model first using Training API."
            }), 500
        
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Please provide 'text' in the request body"
            }), 400
        
        text = data['text']
        
        if not text or len(text.strip()) < 10:
            return jsonify({
                "error": "Text is too short. Please provide at least 10 characters."
            }), 400
        
        # Preprocess
        cleaned_text = preprocess_text(text)
        
        if not cleaned_text:
            return jsonify({
                "error": "Text contains no valid words after preprocessing."
            }), 400
        
        # Vectorize and predict
        text_vectorized = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Get top influential features
        top_features = get_top_features(text_vectorized, prediction, n=10)
        
        # Generate explanation
        explanation = explain_prediction(text, prediction, probabilities, top_features)
        
        # Prepare response
        result = {
            "prediction": "FAKE" if prediction == 1 else "REAL",
            "confidence": {
                "real": float(probabilities[0]),
                "fake": float(probabilities[1])
            },
            "confidence_percentage": {
                "real": f"{probabilities[0]*100:.2f}%",
                "fake": f"{probabilities[1]*100:.2f}%"
            },
            "explanation": explanation,
            "top_words": [
                {
                    "word": word,
                    "influence": "fake" if weight > 0 else "real",
                    "importance": float(importance)
                }
                for word, importance, weight in top_features[:5]
            ]
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("PREDICTION API - Fake News Detection with Explainability")
    print("="*60)
    print("\nAPI will be available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  POST /predict - Predict with explanation")
    print("  GET  /health  - Health check")
    print("\nFeatures:")
    print("  ✓ Real-time prediction")
    print("  ✓ Confidence scores")
    print("  ✓ AI explanation (why fake/real)")
    print("  ✓ Top influential words")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)