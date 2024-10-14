import streamlit as st
from transformers import pipeline
import torch
import os

# Set your Hugging Face token (keep this secure and private)
os.environ["REPLICATE_API_TOKEN"] = "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"

# Set the model ID for Llama 3.2
model_id = "meta-llama/Llama-3.2-1B"

# Load the model with caching for performance
@st.cache_resource
def load_model():
    return pipeline(
        "text-generation", 
        model=model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

# Initialize the model
model = load_model()

# Streamlit app layout
st.title("Llama 3.2 Text Generator")
st.write("Enter your prompt below and get generated text.")

# User input for prompt
user_input = st.text_area("Prompt", "The key to life is")

# Set a slider for controlling the output length
max_length = st.slider("Max Length of Output", min_value=50, max_value=300, value=150, step=10)

if st.button("Generate"):
    if user_input:
        with st.spinner("Generating text..."):
            # Generate text
            generated_text = model(user_input, max_length=max_length, num_return_sequences=1)[0]['generated_text']
            st.text_area("Generated Text", generated_text, height=300)
    else:
        st.error("Please enter a prompt before generating text.")
