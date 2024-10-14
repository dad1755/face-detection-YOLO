import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Set your Hugging Face token (keep this secure and private)
os.environ["REPLICATE_API_TOKEN"] = "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"

# Load the model and tokenizer directly from Hugging Face
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16, device_map="auto")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Initialize the model and tokenizer
tokenizer, model = load_model()

# Streamlit app layout
st.title("Llama 3.2 Text Generator")
st.write("Enter your prompt below and get generated text.")

# User input for prompt
user_input = st.text_area("Prompt", "The key to life is")

# Set a slider for controlling the output length
max_length = st.slider("Max Length of Output", min_value=50, max_value=300, value=150, step=10)

if st.button("Generate"):
    if user_input and model:
        with st.spinner("Generating text..."):
            # Tokenize the input prompt
            input_ids = tokenizer.encode(user_input, return_tensors='pt').to(model.device)

            # Generate text
            generated_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            st.text_area("Generated Text", generated_text, height=300)
    else:
        st.error("Please enter a prompt before generating text.")
