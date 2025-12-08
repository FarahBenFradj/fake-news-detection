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

# Load model and vectorizer with better error handling
@st.cache_resource
def load_model():
    import os
    
    # List all files to debug
    st.write("DEBUG: Looking for model files...")
    if os.path.exists('models'):
        st.write("Files in models/:", os.listdir('models'))
    
    # Try different file locations and names
    model_paths = [
        ('models/best_logistic_regression_model.pkl', 'models/best_tfidf_vectorizer.pkl'),
        ('models/fake_news_model.pkl', 'models/tfidf_vectorizer.pkl'),
        ('best_logistic_regression_model.pkl', 'best_tfidf_vectorizer.pkl'),
        ('fake_news_model.pkl', 'tfidf_vectorizer.pkl'),
    ]
    
    for model_path, vectorizer_path in model_paths:
        try:
            st.write(f"Trying to load: {model_path} and {vectorizer_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            st.success(f"✅ Successfully loaded models from {model_path}")
            return model, vectorizer
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"Failed to load {model_path}: {str(e)}")
            continue
    
    # If all attempts fail
    raise FileNotFoundError("Could not find or load model files. Please check if the model files are in the repository.")

def preprocess_text(text):
    """Preprocess the input text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Tokenization and lemmatization
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
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
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title('🔍 Fake News Detection System')
st.markdown('**Powered by Logistic Regression & Machine Learning**')
st.markdown('</div>', unsafe_allow_html=True)

# Load model
try:
    model, vectorizer = load_model()
except Exception as e:
    st.error(f'❌ Error loading model: {str(e)}')
    st.info("Please make sure the model files are in the repository:")
    st.code("""
    models/
        best_logistic_regression_model.pkl
        best_tfidf_vectorizer.pkl
    """)
    st.stop()

# Input section
st.markdown('---')
st.subheader('📝 Enter News Article or Headline')
st.markdown('*Paste the news text you want to verify below:*')

text_input = st.text_area(
    'News Text',
    height=200,
    placeholder='Example: "Breaking news: Scientists discover new planet..."',
    label_visibility='collapsed'
)

# Add some example texts
with st.expander("📋 Try Example Texts"):
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('Example: Real News'):
            st.session_state.example_text = "Scientists at NASA have confirmed the discovery of water molecules on the Moon's sunlit surface, a breakthrough that could help future lunar missions."
    
    with col2:
        if st.button('Example: Fake News'):
            st.session_state.example_text = "BREAKING: Aliens landed in New York City yesterday and the government is hiding it from everyone! Share before they delete this!"

# Use example text if set
if 'example_text' in st.session_state:
    text_input = st.session_state.example_text
    del st.session_state.example_text
    st.rerun()

# Analyze button
if st.button('🔎 Analyze News', type='primary', use_container_width=True):
    if text_input.strip():
        with st.spinner('🔄 Analyzing the news article...'):
            try:
                # Preprocess text
                processed_text = preprocess_text(text_input)
                
                if not processed_text.strip():
                    st.warning('⚠️ The text is too short or contains no meaningful words. Please provide more content.')
                    st.stop()
                
                # Vectorize
                text_vectorized = vectorizer.transform([processed_text])
                
                # Predict
                prediction = model.predict(text_vectorized)[0]
                probability = model.predict_proba(text_vectorized)[0]
                
                # Display results
                st.markdown('---')
                st.subheader('📊 Analysis Results')
                
                # Main prediction
                if prediction == 1:
                    st.error('### 🚨 FAKE NEWS DETECTED')
                    st.markdown('**This article shows characteristics of fake or misleading news.**')
                    confidence = probability[1] * 100
                    confidence_color = 'red'
                else:
                    st.success('### ✅ REAL NEWS')
                    st.markdown('**This article appears to be legitimate news.**')
                    confidence = probability[0] * 100
                    confidence_color = 'green'
                
                # Confidence metric
                st.markdown(f'**Confidence Level:** :{confidence_color}[{confidence:.1f}%]')
                
                # Progress bar
                st.progress(confidence / 100)
                
                # Detailed probabilities
                st.markdown('---')
                st.subheader('🎯 Detailed Probability Breakdown')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label='Real News',
                        value=f'{probability[0]*100:.1f}%',
                        delta='Legitimate' if prediction == 0 else None
                    )
                
                with col2:
                    st.metric(
                        label='Fake News',
                        value=f'{probability[1]*100:.1f}%',
                        delta='Suspicious' if prediction == 1 else None
                    )
                
                # Additional info
                st.info('💡 **Tip:** This model analyzes text patterns, writing style, and linguistic features to detect fake news. Always verify important news from multiple reliable sources.')
                
            except Exception as e:
                st.error(f'❌ Error during prediction: {str(e)}')
                st.exception(e)
    else:
        st.warning('⚠️ Please enter some text to analyze')

# Footer
st.markdown('---')
st.markdown('''
    <div style="text-align: center; color: gray; font-size: 0.9em;">
        <p><strong>Disclaimer:</strong> This tool uses machine learning and may not be 100% accurate. 
        Always verify news from multiple credible sources.</p>
        <p>Built with Streamlit 🎈 | Model: Logistic Regression</p>
    </div>
''', unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.header('ℹ️ About')
    st.markdown('''
        This fake news detection system uses:
        - **Logistic Regression** for classification
        - **TF-IDF Vectorization** for text processing
        - **NLTK** for natural language processing
        
        ### How it works:
        1. Text is preprocessed (cleaned, lemmatized)
        2. Converted to numerical features using TF-IDF
        3. Classified by the trained model
        4. Confidence score is calculated
        
        ### Tips for best results:
        - Provide complete sentences or paragraphs
        - Include the main content of the article
        - Longer text generally gives better results
    ''')
    
    st.markdown('---')
    st.markdown('**Need help?** Check the examples above!')