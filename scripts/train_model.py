import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load Dataset - Two separate CSV files
print("Loading datasets...")

# Load fake news dataset
fake_df = pd.read_csv('Fake.csv')
fake_df['label'] = 1  # 1 = Fake news

# Load true news dataset
true_df = pd.read_csv('True.csv')
true_df['label'] = 0  # 0 = Real news

# Combine both datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Display basic info
print(f"\nFake news samples: {len(fake_df)}")
print(f"True news samples: {len(true_df)}")
print(f"Total samples: {len(df)}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")

# Text Preprocessing Function
def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    text = re.sub(r'\@\w+|\#\w+', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

# Combine title and text (if both exist)
print("\nPreprocessing text...")
if 'title' in df.columns and 'text' in df.columns:
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
elif 'text' in df.columns:
    df['content'] = df['text'].fillna('')
elif 'title' in df.columns:
    df['content'] = df['title'].fillna('')

# Apply preprocessing
df['cleaned_text'] = df['content'].apply(preprocess_text)

# Remove empty rows
df = df[df['cleaned_text'].str.len() > 0]

print(f"Dataset shape after cleaning: {df.shape}")

# Prepare features and labels
X = df['cleaned_text']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# TF-IDF Vectorization
print("\nVectorizing text with TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Models
print("\n" + "="*50)
print("Training Models...")
print("="*50)

# 1. Logistic Regression
print("\n1. Logistic Regression")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Accuracy: {lr_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, lr_pred, target_names=['Real', 'Fake']))

# 2. Naive Bayes
print("\n2. Multinomial Naive Bayes")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)
nb_accuracy = accuracy_score(y_test, nb_pred)
print(f"Accuracy: {nb_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, nb_pred, target_names=['Real', 'Fake']))

# Select best model
if lr_accuracy >= nb_accuracy:
    best_model = lr_model
    best_name = "Logistic Regression"
    best_accuracy = lr_accuracy
else:
    best_model = nb_model
    best_name = "Naive Bayes"
    best_accuracy = nb_accuracy

print(f"\n{'='*50}")
print(f"Best Model: {best_name} with accuracy {best_accuracy:.4f}")
print(f"{'='*50}")

# Save the model and vectorizer
print("\nSaving model and vectorizer...")
joblib.dump(best_model, 'fake_news_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("Model saved as 'fake_news_model.pkl'")
print("Vectorizer saved as 'tfidf_vectorizer.pkl'")

# Detailed Evaluation
print("\n" + "="*50)
print("DETAILED MODEL EVALUATION")
print("="*50)

# Get predictions on test set
y_pred_test = best_model.predict(X_test_tfidf)
y_pred_proba = best_model.predict_proba(X_test_tfidf)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(f"                Predicted Real    Predicted Fake")
print(f"Actually Real:  {cm[0][0]:>14}    {cm[0][1]:>14}")
print(f"Actually Fake:  {cm[1][0]:>14}    {cm[1][1]:>14}")

# Calculate metrics
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

print(f"\nPrecision (Fake detection): {precision:.4f}")
print(f"Recall (Fake detection): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Test with real samples from test set
print("\n" + "="*50)
print("Sample Predictions from Test Set")
print("="*50)

# Get some real and fake samples from test set
real_samples = X_test[y_test == 0].head(3).tolist()
fake_samples = X_test[y_test == 1].head(3).tolist()

print("\n--- REAL NEWS SAMPLES ---")
for i, text in enumerate(real_samples, 1):
    cleaned = preprocess_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = best_model.predict(vectorized)[0]
    probability = best_model.predict_proba(vectorized)[0]
    
    print(f"\nSample {i}: {text[:100]}...")
    print(f"Actual: REAL | Predicted: {'FAKE' if prediction == 1 else 'REAL'}")
    print(f"Confidence: Real={probability[0]:.2%}, Fake={probability[1]:.2%}")
    if prediction == 0:
        print("✓ CORRECT")
    else:
        print("✗ WRONG")

print("\n--- FAKE NEWS SAMPLES ---")
for i, text in enumerate(fake_samples, 1):
    cleaned = preprocess_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = best_model.predict(vectorized)[0]
    probability = best_model.predict_proba(vectorized)[0]
    
    print(f"\nSample {i}: {text[:100]}...")
    print(f"Actual: FAKE | Predicted: {'FAKE' if prediction == 1 else 'REAL'}")
    print(f"Confidence: Real={probability[0]:.2%}, Fake={probability[1]:.2%}")
    if prediction == 1:
        print("✓ CORRECT")
    else:
        print("✗ WRONG")

# Save model info
model_info = {
    "model_name": best_name,
    "accuracy": float(best_accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "test_size": len(y_test),
    "train_size": len(y_train)
}

import json
with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("\n" + "="*50)
print("✓ Training Complete!")
print("="*50)
print("\nFiles created:")
print("  - fake_news_model.pkl (trained model)")
print("  - tfidf_vectorizer.pkl (text vectorizer)")
print("  - model_info.json (model statistics)")
print(f"\nFinal Accuracy: {best_accuracy:.2%}")
print(f"Model Type: {best_name}")