from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

print("Creating fresh model files...")

# Create models directory
os.makedirs('models', exist_ok=True)

# Training data (expand this with more examples)
texts = [
    "Scientists at NASA announced a major breakthrough in space exploration technology",
    "SHOCKING: Government hiding truth about aliens! Click now before deleted!",
    "Research published in peer-reviewed journal shows climate patterns changing",
    "You won't believe what this celebrity did! Number 7 will shock you!",
    "Economic report indicates steady growth in manufacturing sector",
    "Miracle cure discovered! Doctors hate this one simple trick!",
    "University study reveals new insights into behavioral psychology",
    "Breaking: Illuminati controls everything! Share before censored!",
    "New legislation passed after months of debate in Congress",
    "Amazing secret to lose 50 pounds in one week without exercise!"
]

labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 = real, 1 = fake

# Train vectorizer
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X, labels)

# Save with protocol 4 (more compatible)
with open('models/best_logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=4)

with open('models/best_tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f, protocol=4)

print("✅ Models created successfully!")
print(f"Model file size: {os.path.getsize('models/best_logistic_regression_model.pkl')} bytes")
print(f"Vectorizer file size: {os.path.getsize('models/best_tfidf_vectorizer.pkl')} bytes")