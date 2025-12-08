import gradio as gr
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load model
try:
    model = joblib.load('models/best_logistic_regression_model.joblib')
    vectorizer = joblib.load('models/best_tfidf_vectorizer.joblib')
except:
    # Fallback to pickle if joblib fails
    import pickle
    with open('models/best_logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/best_tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

def preprocess_text(text):
    """Preprocess the input text"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

def predict_news(text):
    """Predict if news is fake or real"""
    if not text.strip():
        return "⚠️ Please enter some text to analyze", 0.0, 0.0
    
    try:
        # Preprocess
        processed_text = preprocess_text(text)
        
        if not processed_text.strip():
            return "⚠️ Text too short or contains no meaningful words", 0.0, 0.0
        
        # Vectorize and predict
        text_vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]
        
        # Format result
        if prediction == 1:
            result = "🚨 **FAKE NEWS DETECTED**"
            confidence = probability[1] * 100
        else:
            result = "✅ **REAL NEWS**"
            confidence = probability[0] * 100
        
        real_prob = probability[0] * 100
        fake_prob = probability[1] * 100
        
        return result, real_prob, fake_prob
        
    except Exception as e:
        return f"❌ Error: {str(e)}", 0.0, 0.0

# Example texts
examples = [
    ["Scientists at NASA have confirmed the discovery of water molecules on the Moon's sunlit surface, a breakthrough that could help future lunar missions."],
    ["BREAKING: Aliens landed in New York City yesterday and the government is hiding it from everyone! Share before they delete this!"],
    ["The stock market reached new heights today as investors responded positively to the Federal Reserve's latest economic policy announcement."],
    ["Doctors discovered that eating ice cream cures cancer! Big pharma doesn't want you to know this secret!"]
]

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Fake News Detector") as demo:
    
    gr.Markdown("""
    # 🔍 Fake News Detection System
    ### Powered by Machine Learning & Logistic Regression
    
    Paste any news article or headline below to check if it's likely to be fake or real news.
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="📝 Enter News Article or Headline",
                placeholder="Paste your news text here...",
                lines=8
            )
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                submit_btn = gr.Button("🔎 Analyze", variant="primary")
    
    with gr.Row():
        with gr.Column():
            output_result = gr.Markdown(label="Result")
            
            with gr.Row():
                real_prob = gr.Number(label="✅ Real News Probability (%)", precision=2)
                fake_prob = gr.Number(label="🚨 Fake News Probability (%)", precision=2)
    
    gr.Markdown("### 📋 Try These Examples:")
    gr.Examples(
        examples=examples,
        inputs=input_text,
        label="Click an example to try it"
    )
    
    gr.Markdown("""
    ---
    **⚠️ Disclaimer:** This tool uses machine learning and may not be 100% accurate. 
    Always verify important news from multiple reliable sources.
    
    **ℹ️ How it works:**
    1. Text is preprocessed (cleaned, lemmatized)
    2. Converted to numerical features using TF-IDF
    3. Classified by a trained Logistic Regression model
    4. Confidence scores are calculated
    """)
    
    # Button actions
    submit_btn.click(
        fn=predict_news,
        inputs=input_text,
        outputs=[output_result, real_prob, fake_prob]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", 0.0, 0.0),
        outputs=[input_text, output_result, real_prob, fake_prob]
    )

# Launch
if __name__ == "__main__":
    demo.launch()