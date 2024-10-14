import streamlit as st
from transformers import pipeline
import os

# Load the Hugging Face token from Streamlit secrets
hf_token = st.secrets["huggingface"]["token"]

# Set the environment variable for huggingface_hub
os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

# Load the model pipeline
classifier = pipeline("sentiment-analysis", use_auth_token=hf_token)

st.title("Sentiment Analysis App")
st.write("Enter text to analyze its sentiment:")

# Input from user
user_input = st.text_area("Text Input", "")

if st.button("Analyze"):
    if user_input:
        # Get the sentiment prediction
        result = classifier(user_input)
        st.write(f"Label: {result[0]['label']}, Score: {result[0]['score']:.2f}")
    else:
        st.write("Please enter some text.")
