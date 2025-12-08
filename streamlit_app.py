import streamlit as st
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pickle

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')

download_nltk_data()

# Load model and vectorizer
@st.cache_resource
def load_model():
    import os
    
    # Try different file locations
    model_paths = [
        ('models/best_logistic_regression_model.pkl', 'models/best_tfidf_vectorizer.pkl'),
        ('models/fake_news_model.pkl', 'models/tfidf_vectorizer.pkl'),
        ('best_logistic_regression_model.pkl', 'best_tfidf_vectorizer.pkl'),
        ('fake_news_model.pkl', 'tfidf_vectorizer.pkl'),
    ]
    
    for model_path, vectorizer_path in model_paths:
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            return model, vectorizer
        except (FileNotFoundError, Exception):
            continue
    
    # If all attempts fail
    raise FileNotFoundError("Could not find model files")

def preprocess_text(text):
    """
    Preprocess text EXACTLY like training code
    Must match training_api.py preprocessing!
    """
    if not text or pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    text = re.sub(r'\@\w+|\#\w+', '', text)
    
    # Remove special characters (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Stop words and lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) 
             for word in words 
             if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title('🔍 Fake News Detection System')
st.markdown('**Powered by Logistic Regression & Machine Learning**')
st.markdown('**Accuracy: 98.63% | F1-Score: 0.9852**')
st.markdown('</div>', unsafe_allow_html=True)

# Load model
try:
    model, vectorizer = load_model()
except Exception as e:
    st.error(f'❌ Error loading model: {str(e)}')
    st.info("Please make sure these files exist:")
    st.code("models/best_logistic_regression_model.pkl\nmodels/best_tfidf_vectorizer.pkl")
    st.stop()

# Input section
st.markdown('---')
st.subheader('📝 Enter News Article')

text_input = st.text_area(
    'News Text',
    height=200,
    placeholder='Paste article text here...',
    label_visibility='collapsed'
)

# Example texts
with st.expander("📋 Try Example Texts"):
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('Real News Example'):
            text_input = "Scientists at Harvard University have published a new study showing that climate change is accelerating. The research, conducted over 10 years, demonstrates significant warming trends."
    
    with col2:
        if st.button('Fake News Example'):
            text_input = "SHOCKING: This amazing secret will blow your mind! Celebrities HATE this one trick! Click now before it's deleted!"

# Analyze button
if st.button('🔎 Analyze News', type='primary', use_container_width=True):
    if text_input.strip():
        with st.spinner('🔄 Analyzing...'):
            try:
                # Preprocess
                processed_text = preprocess_text(text_input)
                
                if not processed_text.strip():
                    st.warning('⚠️ Text too short or no meaningful words. Provide more content.')
                    st.stop()
                
                # Vectorize and predict
                text_vectorized = vectorizer.transform([processed_text])
                prediction = model.predict(text_vectorized)[0]
                probability = model.predict_proba(text_vectorized)[0]
                
                # Display results
                st.markdown('---')
                st.subheader('📊 Results')
                
                if prediction == 1:  # FAKE
                    st.error('### 🚨 LIKELY FAKE NEWS')
                    confidence = probability[1] * 100
                else:  # REAL
                    st.success('### ✅ LIKELY REAL NEWS')
                    confidence = probability[0] * 100
                
                # Confidence
                st.metric('Confidence', f'{confidence:.1f}%')
                st.progress(confidence / 100)
                
                # Detailed breakdown
                st.markdown('---')
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        'Real News Score',
                        f'{probability[0]*100:.1f}%'
                    )
                
                with col2:
                    st.metric(
                        'Fake News Score',
                        f'{probability[1]*100:.1f}%'
                    )
                
                # Info
                st.info('💡 Always verify important news from multiple reliable sources.')
                
            except Exception as e:
                st.error(f'❌ Prediction error: {str(e)}')
    else:
        st.warning('⚠️ Please enter text to analyze')

# Footer
st.markdown('---')
st.markdown('''
    <div style="text-align: center; color: gray; font-size: 0.85em;">
        <p><strong>Disclaimer:</strong> ML model may not be 100% accurate.</p>
        <p>Built with Streamlit | Model: Logistic Regression (TF-IDF)</p>
    </div>
''', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header('ℹ️ How It Works')
    st.markdown('''
        **Model:** Logistic Regression
        
        **Features:** TF-IDF Vectorization
        - 5,000 top features
        - Unigrams + Bigrams
        
        **Preprocessing:**
        1. Lowercase
        2. Remove URLs
        3. Remove special chars
        4. Lemmatization
        5. Stop words removal
        
        **Performance:**
        - Accuracy: 98.63%
        - Precision: 98.97%
        - Recall: 98.07%
    ''')