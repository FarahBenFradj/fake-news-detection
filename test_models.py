# test_models.py
import pickle

print("Testing model loading...")

try:
    with open('models/best_logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded!")
    print(f"Model type: {type(model)}")
    
    with open('models/best_tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Vectorizer loaded!")
    print(f"Vectorizer features: {len(vectorizer.get_feature_names_out())}")
    
    # Test prediction
    test_text = ["This is a test news article"]
    X = vectorizer.transform(test_text)
    pred = model.predict(X)
    print(f"✅ Test prediction works! Result: {pred[0]}")
    
except Exception as e:
    print(f"❌ Error: {e}")