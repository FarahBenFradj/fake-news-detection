import streamlit as st
import requests

st.set_page_config(page_title="🔍 Fake News Detector", layout="wide")

st.title("🔍 Fake News Detection System")
st.markdown("Powered by Logistic Regression & Machine Learning")

st.subheader("📝 Enter News Article or Headline")
text_input = st.text_area("Paste your news text here:", height=150)

if st.button("🔍 Analyze News", use_container_width=True):
    if len(text_input) < 10:
        st.error("❌ Text too short!")
    else:
        try:
            response = requests.post(
                "http://localhost:5000/predict",
                json={"text": text_input},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                prediction = data['prediction']
                confidence_real = data['confidence_percentage']['real']
                confidence_fake = data['confidence_percentage']['fake']
                
                if prediction == "REAL":
                    st.success(f"## ✅ REAL NEWS")
                else:
                    st.error(f"## ⚠️ FAKE NEWS")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Real News Confidence", confidence_real)
                with col2:
                    st.metric("Fake News Confidence", confidence_fake)
                
                st.markdown("### 💡 Explanation")
                explanation = data.get('explanation', {})
                st.write(f"**Verdict:** {explanation.get('verdict')}")
                st.write(f"**Reasons:** {explanation.get('summary')}")
            else:
                st.error(f"Error: {response.json().get('error')}")
        except Exception as e:
            st.error(f"❌ Cannot connect to API: {str(e)}")