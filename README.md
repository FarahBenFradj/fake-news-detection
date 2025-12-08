# 🔍 Fake News Detection System

A machine learning system for detecting fake news.

## 📊 Performance
- **Logistic Regression:** 98.63% accuracy
- **Naive Bayes:** 95.2% accuracy

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/compare_models.py
python scripts/hyperparameter_tuning.py
streamlit run api/streamlit_app.py
```

## 📁 Structure
- `scripts/` - Training scripts
- `api/` - APIs
- `data/` - Datasets
- `models/` - Trained models
- `frontend/` - React app

## 🏆 Results
- Accuracy: 98.63%
- F1-Score: 0.9852
- Inference: <50ms