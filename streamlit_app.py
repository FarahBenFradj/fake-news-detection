import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Load model and vectorizer
@st.cache_resource
def load_model():
    with open('model/best_logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def preprocess_text(text):
    """Preprocess the input text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenization and lemmatization
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Streamlit UI
st.title('🔍 Fake News Detection System')
st.markdown('**Powered by Logistic Regression & Machine Learning**')

# Load model
try:
    model, vectorizer = load_model()
    st.success('✅ Model loaded successfully!')
except Exception as e:
    st.error(f'❌ Error loading model: {str(e)}')
    st.stop()

# Input section
st.subheader('📝 Enter News Article or Headline')
text_input = st.text_area('Paste your news text here:', height=200)

if st.button('🔎 Analyze', type='primary'):
    if text_input.strip():
        with st.spinner('Analyzing...'):
            try:
                # Preprocess text
                processed_text = preprocess_text(text_input)
                
                # Vectorize
                text_vectorized = vectorizer.transform([processed_text])
                
                # Predict
                prediction = model.predict(text_vectorized)[0]
                probability = model.predict_proba(text_vectorized)[0]
                
                # Display results
                st.markdown('---')
                st.subheader('📊 Analysis Results')
                
                if prediction == 1:
                    st.error('🚨 **FAKE NEWS DETECTED**')
                    confidence = probability[1] * 100
                else:
                    st.success('✅ **REAL NEWS**')
                    confidence = probability[0] * 100
                
                st.metric('Confidence', f'{confidence:.2f}%')
                
                # Show probability distribution
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('Real News Probability', f'{probability[0]*100:.2f}%')
                with col2:
                    st.metric('Fake News Probability', f'{probability[1]*100:.2f}%')
                    
            except Exception as e:
                st.error(f'❌ Error during prediction: {str(e)}')
    else:
        st.warning('⚠️ Please enter some text to analyze')

# Add footer
st.markdown('---')
st.markdown('*Disclaimer: This tool uses machine learning and may not be 100% accurate.*')