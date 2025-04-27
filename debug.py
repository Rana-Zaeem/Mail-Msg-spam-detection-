import streamlit as st
import os
import pickle

st.title("Debug Information")

# List all files in the current directory
st.write("### Files in deployment directory:")
files_list = os.listdir(".")
for f in files_list:
    st.write(f"- {f} ({os.path.getsize(f)/1024:.1f} KB)")

# Check if model files exist
model_path = 'model.pkl'
vectorizer_path = 'vectorizer.pkl'

st.write("### Model file checks:")
if os.path.exists(model_path):
    st.success(f"✅ Model file exists at: {os.path.abspath(model_path)}")
    st.write(f"Size: {os.path.getsize(model_path)/1024:.1f} KB")
    
    # Try to load the model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully")
        st.write(f"Model type: {type(model)}")
    except Exception as e:
        st.error(f"❌ Could not load model: {str(e)}")
else:
    st.error(f"❌ Model file not found at: {os.path.abspath(model_path)}")

if os.path.exists(vectorizer_path):
    st.success(f"✅ Vectorizer file exists at: {os.path.abspath(vectorizer_path)}")
    st.write(f"Size: {os.path.getsize(vectorizer_path)/1024:.1f} KB")
    
    # Try to load the vectorizer
    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        st.success("✅ Vectorizer loaded successfully")
        st.write(f"Vectorizer type: {type(vectorizer)}")
    except Exception as e:
        st.error(f"❌ Could not load vectorizer: {str(e)}")
else:
    st.error(f"❌ Vectorizer file not found at: {os.path.abspath(vectorizer_path)}")

# Check environment variables
st.write("### Environment variables:")
for key, val in os.environ.items():
    if key.startswith(('PYTHONPATH', 'PATH', 'STREAMLIT')):
        st.code(f"{key}: {val}")

# Show Python information
import sys
st.write("### Python information:")
st.code(f"Python version: {sys.version}")
st.code(f"Python executable: {sys.executable}")
st.code(f"Python path: {sys.path}")

st.write("### Memory usage:")
import psutil
process = psutil.Process(os.getpid())
st.code(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")