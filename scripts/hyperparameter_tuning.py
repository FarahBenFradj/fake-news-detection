import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

print("="*80)
print("HYPERPARAMETER TUNING - LOGISTIC REGRESSION")
print("="*80)

# Step 1: Load data
print("\n[1/5] Loading data...")
try:
    fake_df = pd.read_csv('Fake.csv')
    true_df = pd.read_csv('True.csv')
    print(f"✓ Loaded {len(fake_df)} fake + {len(true_df)} true articles")
except FileNotFoundError:
    print("❌ Error: Fake.csv or True.csv not found!")
    exit()

# Label data
fake_df['label'] = 1
true_df['label'] = 0

# Combine and shuffle
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"✓ Total: {len(df)} articles")

# Step 2: Preprocessing function
def preprocess_text(text):
    """Clean and preprocess text"""
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

# Step 3: Preprocess text
print("\n[2/5] Preprocessing text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)
df = df[df['cleaned_text'].str.len() > 0]
print(f"✓ Processed {len(df)} articles")

X = df['cleaned_text']
y = df['label']

# Step 4: Split data
print("\n[3/5] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Training: {len(X_train)} samples")
print(f"✓ Test: {len(X_test)} samples")

# Step 5: Vectorize
print("\n[4/5] Vectorizing text with TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"✓ Created {X_train_tfidf.shape[1]} features")

# Step 6: Hyperparameter tuning with GridSearchCV
print("\n[5/5] Tuning hyperparameters with GridSearchCV...")
print("-" * 80)

param_grid = {
    'C': [0.1, 1, 10, 100],              # Regularization strength
    'max_iter': [1000, 2000],            # Max iterations
    'penalty': ['l2'],                   # Penalty type
    'solver': ['lbfgs', 'liblinear']     # Solver algorithm
}

print(f"\nSearching grid with {len(param_grid['C']) * len(param_grid['max_iter']) * len(param_grid['penalty']) * len(param_grid['solver'])} combinations...")
print(f"Parameters: {param_grid}\n")

# Create GridSearchCV
gs = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring='f1',            # Optimize for F1-score
    n_jobs=-1,               # Use all cores
    verbose=1
)

# Fit
gs.fit(X_train_tfidf, y_train)

print("\n" + "="*80)
print("TUNING RESULTS")
print("="*80)

print(f"\n✓ Best Parameters Found:")
for param, value in gs.best_params_.items():
    print(f"   {param}: {value}")

print(f"\n✓ Best CV Score (F1): {gs.best_score_:.4f}")

# Evaluate on test set
best_model = gs.best_estimator_
y_pred = best_model.predict(X_test_tfidf)

test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print(f"\n✓ Test Set Performance:")
print(f"   Accuracy: {test_accuracy:.4f}")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall: {test_recall:.4f}")
print(f"   F1-Score: {test_f1:.4f}")

# Compare with baseline
print("\n" + "="*80)
print("COMPARISON: BASELINE vs TUNED")
print("="*80)

baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train_tfidf, y_train)
baseline_pred = baseline_model.predict(X_test_tfidf)

baseline_accuracy = accuracy_score(y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred)

improvement_accuracy = (test_accuracy - baseline_accuracy) * 100
improvement_f1 = (test_f1 - baseline_f1) * 100

print(f"\nBaseline (default params):")
print(f"   Accuracy: {baseline_accuracy:.4f}")
print(f"   F1-Score: {baseline_f1:.4f}")

print(f"\nTuned (GridSearchCV):")
print(f"   Accuracy: {test_accuracy:.4f}")
print(f"   F1-Score: {test_f1:.4f}")

print(f"\nImprovement:")
print(f"   Accuracy: {improvement_accuracy:+.4f}%")
print(f"   F1-Score: {improvement_f1:+.4f}%")

# Save best model
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

joblib.dump(best_model, 'best_logistic_regression_model.pkl')
joblib.dump(tfidf, 'best_tfidf_vectorizer.pkl')

print("\n✓ Best model saved: best_logistic_regression_model.pkl")
print("✓ Vectorizer saved: best_tfidf_vectorizer.pkl")

# Save summary
summary = {
    'best_params': gs.best_params_,
    'best_cv_score': float(gs.best_score_),
    'test_accuracy': float(test_accuracy),
    'test_precision': float(test_precision),
    'test_recall': float(test_recall),
    'test_f1': float(test_f1),
    'baseline_accuracy': float(baseline_accuracy),
    'baseline_f1': float(baseline_f1),
    'improvement_accuracy': float(improvement_accuracy),
    'improvement_f1': float(improvement_f1)
}

import json
with open('hyperparameter_tuning_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✓ Results saved: hyperparameter_tuning_results.json")

print("\n" + "="*80)
print("✓ HYPERPARAMETER TUNING COMPLETE!")
print("="*80)