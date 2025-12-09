import streamlit as st
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pickle
import joblib
import os
import requests
import json
import random

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

# Load test samples from GitHub
@st.cache_resource
def load_test_samples():
    """Load test samples from your GitHub repository"""
    try:
        url = "https://raw.githubusercontent.com/FarahBenFradj/fake-news-detection/main/results/test_samples.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        test_samples = response.json()
        return test_samples
    except Exception as e:
        st.warning(f"Could not load test samples from GitHub: {e}")
        return None

# Load model and vectorizer
@st.cache_resource
def load_model():
    """Load the trained model and vectorizer from GitHub Release"""
    
    # URLs from your GitHub Release
    MODEL_URL = "https://github.com/FarahBenFradj/fake-news-detection/releases/download/v1.0.0/best_logistic_regression_model.pkl"
    VECTORIZER_URL = "https://github.com/FarahBenFradj/fake-news-detection/releases/download/v1.0.0/best_tfidf_vectorizer.pkl"
    
    # Create models folder if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/best_logistic_regression_model.pkl'
    vectorizer_path = 'models/best_tfidf_vectorizer.pkl'
    
    # Download model if not present
    if not os.path.exists(model_path):
        st.info("📥 Downloading model from GitHub Release...")
        try:
            response = requests.get(MODEL_URL, allow_redirects=True, timeout=30)
            response.raise_for_status()
            
            # Check if it's HTML (error page)
            if response.content[:15].lower().startswith(b'<!doctype html') or response.content[:6].lower() == b'<html>':
                raise Exception("GitHub returned HTML instead of pickle file")
            
            # Save file
            with open(model_path, 'wb') as f:
                f.write(response.content)
            
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            if os.path.exists(model_path):
                os.remove(model_path)
            raise
    
    # Download vectorizer if not present
    if not os.path.exists(vectorizer_path):
        st.info("📥 Downloading vectorizer from GitHub Release...")
        try:
            response = requests.get(VECTORIZER_URL, allow_redirects=True, timeout=30)
            response.raise_for_status()
            
            # Check if it's HTML
            if response.content[:15].lower().startswith(b'<!doctype html') or response.content[:6].lower() == b'<html>':
                raise Exception("GitHub returned HTML instead of pickle file")
            
            with open(vectorizer_path, 'wb') as f:
                f.write(response.content)
            
            st.success("✅ Vectorizer downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download vectorizer: {e}")
            if os.path.exists(vectorizer_path):
                os.remove(vectorizer_path)
            raise
    
    # Load the files - try multiple methods
    try:
        # Method 1: Try with joblib (best for sklearn)
        try:
            st.info("🔄 Loading models with joblib...")
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            st.success("✅ Models loaded with joblib!")
            return model, vectorizer
        except Exception as e1:
            st.warning(f"Joblib failed: {str(e1)[:100]}")
            
            # Method 2: Try standard pickle
            try:
                st.info("🔄 Trying standard pickle...")
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                st.success("✅ Models loaded with pickle!")
                return model, vectorizer
            except Exception as e2:
                st.warning(f"Standard pickle failed: {str(e2)[:100]}")
                
                # Method 3: Try with encoding
                try:
                    st.info("🔄 Trying with latin1 encoding...")
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f, encoding='latin1')
                    with open(vectorizer_path, 'rb') as f:
                        vectorizer = pickle.load(f, encoding='latin1')
                    st.success("✅ Models loaded with encoding!")
                    return model, vectorizer
                except Exception as e3:
                    st.error(f"All loading methods failed!")
                    st.error(f"Error: {str(e3)}")
                    raise e3
                    
    except Exception as e:
        st.error(f"❌ Cannot load model files: {e}")
        st.error("⚠️ Your pickle files may have been created with an incompatible Python/scikit-learn version")
        
        # Show detailed error info
        st.error("**Debugging Info:**")
        st.code(f"Python version on server: {st.session_state.get('python_version', 'unknown')}")
        st.code(f"Error type: {type(e).__name__}")
        st.code(f"Error message: {str(e)}")
        
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(vectorizer_path):
            os.remove(vectorizer_path)
        raise

def preprocess_text(text):
    """
    Preprocess text EXACTLY matching training_api.py
    This is CRITICAL for correct predictions!
    """
    if not text or pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions (@user) and hashtags (#hashtag)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    
    # Remove special characters - keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization, stop words removal, and lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) 
             for word in words 
             if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🔍 Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .real-news {
        border-left: 4px solid #2ecc71;
    }
    .fake-news {
        border-left: 4px solid #e74c3c;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title('🔍 Fake News Detection System')
st.markdown('**Powered by Logistic Regression & Machine Learning**')
st.markdown('**Accuracy: 98.63% | F1-Score: 0.9852**')
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================

try:
    model, vectorizer = load_model()
except Exception as e:
    st.error(f'❌ Error loading model: {str(e)}')
    st.error("Model files not found!")
    st.info("""
    Please make sure these files exist in your GitHub Release:
    
    - best_logistic_regression_model.pkl
    - best_tfidf_vectorizer.pkl
    """)
    st.stop()

# ============================================================================
# LOAD TEST SAMPLES
# ============================================================================

test_samples = load_test_samples()

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('---')

# Two column layout
col1, col2 = st.columns([2, 1])

# Initialize session state for text input
if 'text_input' not in st.session_state:
    st.session_state['text_input'] = ""

with col1:
    st.subheader('📝 Enter News Article or Headline')
    text_input = st.text_area(
        'News Text',
        value=st.session_state['text_input'],
        height=200,
        placeholder='Paste your news article, headline, or statement here...',
        label_visibility='collapsed',
        key='main_text_input'
    )
    # Update session state when text changes
    st.session_state['text_input'] = text_input

with col2:
    st.subheader('📋 Example Texts')
    
    # Load examples from test_samples.json
    if test_samples:
        if st.button('📰 Real News Example', use_container_width=True):
            # Get a random real news sample
            real_samples = test_samples.get('real_samples', [])
            if real_samples:
                sample = random.choice(real_samples)
                st.session_state['text_input'] = sample.get('full_text', '')
                st.rerun()
        
        if st.button('⚠️ Fake News Example', use_container_width=True):
            # Get a random fake news sample
            fake_samples = test_samples.get('fake_samples', [])
            if fake_samples:
                sample = random.choice(fake_samples)
                st.session_state['text_input'] = sample.get('full_text', '')
                st.rerun()
    else:
        # Fallback examples if test_samples.json can't be loaded
        if st.button('📰 Real News Example', use_container_width=True):
            st.session_state['text_input'] = "Government announces new renewable energy initiative. Scientists from leading universities conducted extensive research showing climate change effects."
            st.rerun()
        
        if st.button('⚠️ Fake News Example', use_container_width=True):
            st.session_state['text_input'] = "SHOCKING! This amazing secret will blow your mind! You won't believe what doctors don't want you to know! Click here before it's deleted!"
            st.rerun()

# ============================================================================
# ANALYSIS
# ============================================================================

st.markdown('---')

if st.button('🔎 Analyze News', type='primary', use_container_width=True, key='analyze_btn'):
    if not text_input.strip():
        st.warning('⚠️ Please enter some text to analyze')
    else:
        with st.spinner('🔄 Analyzing the article...'):
            try:
                # Preprocess text
                processed_text = preprocess_text(text_input)
                
                if not processed_text.strip():
                    st.warning('⚠️ The text is too short or contains no meaningful words after processing.')
                    st.info('Try adding more content to your text.')
                else:
                    # Vectorize
                    text_vectorized = vectorizer.transform([processed_text])
                    
                    # Predict
                    prediction = model.predict(text_vectorized)[0]
                    probabilities = model.predict_proba(text_vectorized)[0]
                    
                    prob_real = probabilities[0]
                    prob_fake = probabilities[1]
                    
                    # Display results
                    st.markdown('---')
                    st.subheader('📊 Analysis Results')
                    
                    # Main prediction
                    if prediction == 1:  # FAKE NEWS
                        st.error('### 🚨 LIKELY FAKE NEWS')
                        st.markdown(
                            'This article shows patterns and characteristics typical of fake or misleading news.'
                        )
                        confidence = prob_fake * 100
                    else:  # REAL NEWS
                        st.success('### ✅ LIKELY REAL NEWS')
                        st.markdown(
                            'This article appears to be legitimate news based on linguistic and textual analysis.'
                        )
                        confidence = prob_real * 100
                    
                    # Confidence display
                    st.markdown('---')
                    
                    # Three columns for metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric(
                            label='Primary Confidence',
                            value=f'{confidence:.1f}%',
                            delta='Strong' if confidence > 85 else 'Moderate' if confidence > 70 else 'Weak'
                        )
                    
                    with metric_col2:
                        st.metric(
                            label='Real News Score',
                            value=f'{prob_real*100:.1f}%'
                        )
                    
                    with metric_col3:
                        st.metric(
                            label='Fake News Score',
                            value=f'{prob_fake*100:.1f}%'
                        )
                    
                    # Confidence bar
                    st.markdown('**Confidence Distribution:**')
                    col1, col2 = st.columns([prob_real, prob_fake], gap='small')
                    with col1:
                        st.metric('Real', f'{prob_real*100:.1f}%', label_visibility='collapsed')
                    with col2:
                        st.metric('Fake', f'{prob_fake*100:.1f}%', label_visibility='collapsed')
                    
                    st.progress(max(prob_real, prob_fake))
                    
                    # Information box
                    st.markdown('---')
                    st.info(
                        '💡 **Remember:** Always verify news from multiple credible sources. '
                        'This tool analyzes text patterns but is not 100% accurate.'
                    )
                    
            except Exception as e:
                st.error(f'❌ Error during analysis: {str(e)}')
                st.exception(e)

# ============================================================================
# SIDEBAR - HOW IT WORKS
# ============================================================================

with st.sidebar:
    st.header('ℹ️ About This Tool')
    
    st.subheader('🤖 Model Details')
    st.markdown('''
        **Algorithm:** Logistic Regression
        
        **Feature Extraction:** TF-IDF
        - Maximum Features: 5,000
        - N-gram Range: 1-2 (unigrams + bigrams)
        - Min Document Frequency: 2
    ''')
    
    st.subheader('🔍 How It Works')
    st.markdown('''
        1. **Preprocessing:** Text is cleaned and normalized
        2. **Lemmatization:** Words reduced to base form
        3. **Vectorization:** Text converted to TF-IDF features
        4. **Classification:** Logistic Regression predicts
        5. **Confidence:** Probability score calculated
    ''')
    
    st.subheader('📈 Model Performance')
    st.markdown(f'''
        - **Accuracy:** 98.63%
        - **Precision:** 98.97%
        - **Recall:** 98.07%
        - **F1-Score:** 0.9852
        - **Inference Speed:** <50ms
    ''')
    
    st.subheader('⚙️ Preprocessing Steps')
    st.markdown('''
        1. Convert to lowercase
        2. Remove URLs and links
        3. Remove mentions and hashtags
        4. Remove special characters
        5. Remove stop words (the, a, is, etc.)
        6. Lemmatization (running → run)
        7. Keep only words longer than 2 chars
    ''')
    
    st.markdown('---')
    st.markdown(
        '**Built with:** Streamlit | **Model:** Scikit-learn | **NLP:** NLTK'
    )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown('---')
st.markdown('''
    <div style="text-align: center; color: gray; font-size: 0.85em; padding: 1rem;">
        <p><strong>⚖️ Disclaimer:</strong> This machine learning model may not be 100% accurate. 
        Always verify important news from multiple reliable and credible sources.</p>
        <p>Fake News Detection System | Powered by ML</p>
    </div>
''', unsafe_allow_html=True)