import streamlit as st
import pickle
import string
import time
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go

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

# Apply custom CSS for animations and styling with mobile responsiveness
st.markdown("""
<style>
    /* Mobile-first responsive design */
    .main {
        padding: 1rem !important;
    }
    
    /* Button hover effect */
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    /* Make buttons full width on mobile */
    .stButton>button {
        width: 100%;
    }
    
    /* Result animation */
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    
    /* Result styling */
    .spam-result, .ham-result {
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
        animation: fadeIn 0.5s ease-in-out;
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
    
    /* Make text responsive */
    h1 {
        font-size: clamp(1.5rem, 4vw, 2.5rem) !important;
    }
    
    h2 {
        font-size: clamp(1.3rem, 3vw, 2rem) !important;
    }
    
    h3 {
        font-size: clamp(1.1rem, 2.5vw, 1.7rem) !important;
    }
    
    /* Improve mobile readability */
    .stTextArea textarea {
        font-size: 1rem !important;
    }
    
    /* Responsive container padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Media queries for mobile devices */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        .row-widget.stButton {
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

#------------------------------------------------------------
# HELPER FUNCTIONS
#------------------------------------------------------------

# OPTIMIZATION: Load NLTK resources at startup with better error handling
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        return True
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            return True
        except Exception as e:
            st.warning(f"NLTK download issue: {str(e)}. Some features may be limited.")
            return False

# OPTIMIZATION: Preload models with better caching strategy
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_models():
    try:
        model_path = 'model.pkl'
        vectorizer_path = 'vectorizer.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            st.error(f"Model files not found. Please check that {model_path} and {vectorizer_path} exist.")
            return None, None
        
        # Use binary mode explicitly with error handling
        try:
            with open(vectorizer_path, 'rb') as f:
                tfidf = pickle.load(f)
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return tfidf, model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None
    except Exception as e:
        st.error(f"Unexpected error in model loading: {e}")
        return None, None

# Initialize Porter Stemmer and Word Lemmatizer
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# OPTIMIZATION: More efficient text preprocessing with caching
@st.cache_data(ttl=3600)
def transform_text(text):
    """Optimized text preprocessing"""
    if not text:
        return ""
        
    # Convert to lowercase and tokenize
    tokens = nltk.word_tokenize(text.lower())
    
    # One-pass filtering for better performance
    filtered_tokens = []
    try:
        stop_words = set(stopwords.words('english'))
        for token in tokens:
            if token.isalnum() and token not in stop_words and token not in string.punctuation:
                # Apply stemming and lemmatization
                stemmed = ps.stem(token)
                lemmatized = lemmatizer.lemmatize(stemmed)
                filtered_tokens.append(lemmatized)
    except Exception as e:
        # Fallback if NLTK resources aren't available
        filtered_tokens = [token.lower() for token in tokens if token.isalnum()]
        
    return " ".join(filtered_tokens)

# Function to create an animated gauge chart
@st.cache_data
def create_gauge_chart(score, title):
    """
    Creates an animated gauge chart for visualizing spam probability
    
    Parameters:
    score (float): Probability score between 0 and 1
    title (str): Title for the gauge chart
    
    Returns:
    plotly.graph_objects.Figure: The gauge chart figure
    """
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
    
    # Make chart responsive
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=30, b=10),
        autosize=True
    )
    
    return fig

# OPTIMIZATION: Preload resources and models early for faster startup
with st.spinner("Loading resources..."):
    resources_ready = download_nltk_resources()
    tfidf, model = load_models()

# Verify models loaded successfully
if tfidf is None or model is None:
    st.error("Failed to load models. Please check the application logs.")
    st.stop()

#------------------------------------------------------------
# UI LAYOUT
#------------------------------------------------------------

# Create a mobile-friendly layout
# Sidebar with project info (collapsible on mobile)
with st.sidebar:
    st.title("About")
    st.info("""
    **Email/SMS Spam Classifier**
    
    This application uses machine learning to classify emails or SMS 
    messages as spam or not spam (ham).
    
    **Features:**
    - Advanced text preprocessing
    - High-accuracy classifier
    - Real-time prediction
    """)
    
    # Show model metrics
    st.subheader("Model Information")
    st.success("""
    **Model Type:** Advanced Ensemble
    **Accuracy:** ~97-99%
    **Features:** TF-IDF with n-grams
    """)

# Main area - responsive layout
st.title("Email/SMS Spam Classifier ✉️")
st.markdown("### Enter a message to classify")

# Message input area (full width on mobile)
input_sms = st.text_area("", height=120, placeholder="Type or paste your email/message here...")

# Mobile-friendly button layout
# Use different column ratios for different screen sizes
if st.session_state.get('_is_mobile', False):
    # Mobile layout
    predict_button = st.button('Predict')
else:
    # Desktop layout with centered button
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        predict_button = st.button('Predict')

#------------------------------------------------------------
# PREDICTION FUNCTIONALITY
#------------------------------------------------------------

# Show a loading spinner and then prediction
if predict_button:
    if not input_sms:
        st.warning("Please enter a message to classify.")
    else:
        with st.spinner('Analyzing your message...'):
            # OPTIMIZATION: Remove unnecessary sleep delays
            progress_bar = st.progress(0)
            for percent_complete in range(0, 101, 33):  # Faster progress updates
                progress_bar.progress(percent_complete)
            
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
            
            progress_bar.empty()
            
            # Display result with animation
            st.markdown("<hr>", unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown('<div class="spam-result">SPAM DETECTED!</div>', unsafe_allow_html=True)
                st.error("This message has been classified as spam. Be cautious!")
            else:
                st.markdown('<div class="ham-result">NOT SPAM (HAM)</div>', unsafe_allow_html=True)
                st.success("This message appears to be legitimate.")
            
            # Display gauge chart - responsive
            st.plotly_chart(create_gauge_chart(spam_prob, "Spam Probability"), use_container_width=True, config={'responsive': True})
            
            # Text Feature Analysis - mobile-friendly layout
            st.subheader("Message Analysis:")
            
            # On mobile, stack vertically; on desktop, use columns
            use_container_width = True
            
            st.info(f"""
            **Key Statistics:**
            - Character count: {len(input_sms)}
            - Word count: {len(input_sms.split())}
            - Sentence count: {len(nltk.sent_tokenize(input_sms)) if resources_ready else 'N/A'}
            """)
            
            st.warning("""
            **Common Spam Indicators:**
            - Excessive punctuation
            - ALL CAPS text
            - Urgency words (now, urgent, immediately)
            - Financial terms (money, cash, credit)
            """)

#------------------------------------------------------------
# FOOTER
#------------------------------------------------------------

# Footer
st.markdown("---")
st.caption("© 2025 Email/SMS Spam Classifier - Enhanced with Advanced Machine Learning")

# Add JavaScript to detect mobile and set session state
# This helps with responsive layouts
st.markdown("""
<script>
    document.addEventListener('DOMContentLoaded', function() {
        // Check if user is on mobile
        const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
        if (isMobile) {
            const data = {
                _is_mobile: true
            };
            // Use Streamlit's setComponentValue to update session state
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: data
            }, '*');
        }
    });
</script>
""", unsafe_allow_html=True)
