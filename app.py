import streamlit as st
from transformers import pipeline

# Load the sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis")

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
