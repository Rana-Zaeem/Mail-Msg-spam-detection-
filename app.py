import streamlit as st
import pickle
import string
import nltk
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go

# Force NLTK downloads to ensure they're available (important for cloud deployment)
try:
    # Directly download NLTK resources without checking if they exist
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    st.warning(f"NLTK download issue: {str(e)}. Some features may be limited.")

#------------------------------------------------------------
# CONFIGURATION
#------------------------------------------------------------

# Set page configuration - do this first before any other st calls
st.set_page_config(
    page_title="Email/SMS Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapse sidebar by default on mobile
)

# Apply simplified CSS for better performance while maintaining key styling
st.markdown("""
<style>
    .main {
        padding: 1rem !important;
    }
    .spam-result, .ham-result {
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    .spam-result {
        background-color: rgba(255, 0, 0, 0.1);
        color: #d62728;
        border: 2px solid #d62728;
    }
    .ham-result {
        background-color: rgba(0, 128, 0, 0.1);
        color: #2ca02c;
        border: 2px solid #2ca02c;
    }
</style>
""", unsafe_allow_html=True)

#------------------------------------------------------------
# HELPER FUNCTIONS - IMPROVED FOR STABILITY
#------------------------------------------------------------

# Initialize stemmer and lemmatizer - moved outside functions for single initialization
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Initialize stopwords once to avoid repeated loading
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    # Fallback if stopwords aren't available
    stop_words = set()
    st.warning("Could not load stopwords. Using simplified text processing.")

# Changed caching decorator to be compatible with different Streamlit versions
@st.cache_data(ttl=3600)
def load_models():
    """Load models with better error handling and feedback"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
        vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
        
        # Check if files exist
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None, None
            
        if not os.path.exists(vectorizer_path):
            st.error(f"Vectorizer file not found at: {vectorizer_path}")
            return None, None
        
        # Load the files with better error handling
        with open(vectorizer_path, 'rb') as f:
            tfidf = pickle.load(f)
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        return tfidf, model
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

@st.cache_data(ttl=3600)
def transform_text(text):
    """Text preprocessing with simplified logic for better reliability"""
    if not text:
        return ""
    
    # Simplified implementation that's less likely to fail
    try:
        # Convert to lowercase and tokenize
        tokens = nltk.word_tokenize(text.lower())
        
        # Filter tokens (simplified logic)
        filtered_tokens = []
        for token in tokens:
            if token.isalnum() and token not in stop_words:
                filtered_tokens.append(ps.stem(token))
        
        return " ".join(filtered_tokens)
    except Exception as e:
        # Ultra-fallback if anything fails
        st.warning(f"Error in text processing: {str(e)}. Using basic processing.")
        words = text.lower().split()
        return " ".join([w for w in words if len(w) > 2])

@st.cache_data
def create_gauge_chart(score, title):
    """Creates a gauge chart for visualizing spam probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score*100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#19d3f3" if score < 0.5 else "#f25829"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0, 250, 0, 0.25)'},
                {'range': [50, 100], 'color': 'rgba(250, 0, 0, 0.25)'}
            ],
        }
    ))
    
    # Simplified layout for better performance
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=30, b=10),
        autosize=True
    )
    
    return fig

#------------------------------------------------------------
# UI LAYOUT - OPTIMIZED FOR FASTER LOADING
#------------------------------------------------------------

# Main area - responsive layout
st.title("Email/SMS Spam Classifier ✉️")
st.markdown("### Enter a message to classify")

# Message input area (full width on mobile)
input_sms = st.text_area("", height=120, placeholder="Type or paste your email/message here...")

# Simplified button layout
predict_button = st.button('Predict', use_container_width=True)

# Sidebar with project info - moved after main content for faster initial load
with st.sidebar:
    st.title("About")
    st.info("""
    **Email/SMS Spam Classifier**
    
    This application uses machine learning to classify emails or SMS 
    messages as spam or not spam (ham).
    """)
    
    # Show model metrics
    st.subheader("Model Information")
    st.success("""
    **Model Type:** Advanced Ensemble
    **Accuracy:** ~97-99%
    """)

#------------------------------------------------------------
# PREDICTION FUNCTIONALITY - WITH BETTER ERROR HANDLING
#------------------------------------------------------------

# Pre-load models to check if they work
tfidf, model = None, None
try:
    with st.spinner("Loading models..."):
        tfidf, model = load_models()
except Exception as e:
    st.error(f"Failed to initialize models: {str(e)}")

# Show a loading spinner and then prediction
if predict_button:
    if not input_sms:
        st.warning("Please enter a message to classify.")
    elif tfidf is None or model is None:
        st.error("Models couldn't be loaded. Please check the application logs.")
    else:
        try:
            with st.spinner('Analyzing your message...'):
                # Preprocess
                transformed_sms = transform_text(input_sms)
                
                # Vectorize
                vector_input = tfidf.transform([transformed_sms])
                
                # Predict
                prediction = model.predict(vector_input)[0]
                
                # Get probability scores if the model supports it
                try:
                    probability = model.predict_proba(vector_input)[0]
                    spam_prob = probability[1]
                except:
                    spam_prob = float(prediction)
                
                # Display result without animation for faster rendering
                st.markdown("<hr>", unsafe_allow_html=True)
                
                if prediction == 1:
                    st.markdown('<div class="spam-result">SPAM DETECTED!</div>', unsafe_allow_html=True)
                    st.error("This message has been classified as spam. Be cautious!")
                else:
                    st.markdown('<div class="ham-result">NOT SPAM (HAM)</div>', unsafe_allow_html=True)
                    st.success("This message appears to be legitimate.")
                
                # Display gauge chart - responsive
                st.plotly_chart(create_gauge_chart(spam_prob, "Spam Probability"), use_container_width=True)
                
                # Text Feature Analysis - simplified for performance
                st.subheader("Message Analysis:")
                
                st.info(f"""
                **Key Statistics:**
                - Character count: {len(input_sms)}
                - Word count: {len(input_sms.split())}
                """)
                
                st.warning("""
                **Common Spam Indicators:**
                - Excessive punctuation
                - ALL CAPS text
                - Urgency words (now, urgent, immediately)
                """)
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

#------------------------------------------------------------
# FOOTER
#------------------------------------------------------------

# Footer - simplified
st.markdown("---")
st.caption("© 2025 Email/SMS Spam Classifier")
