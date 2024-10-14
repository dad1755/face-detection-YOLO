import os
import streamlit as st
from transformers import pipeline

# Set your API token for Hugging Face authentication
REPLICATE_API_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = REPLICATE_API_TOKEN

# Initialize the text generation pipeline
try:
    # Use the model with the pipeline
    pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")
except OSError as e:
    st.error("Could not load the model. Please check the model name and your internet connection.")
    st.error(str(e))
    
    # Fallback to a different model (GPT-2)
    st.warning("Falling back to GPT-2 model.")
    pipe = pipeline("text-generation", model="gpt2")

# Streamlit UI
st.title("Llama-3.2 Chatbot")
user_input = st.text_input("You: ", "")

if st.button("Send"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Generate output using the pipeline
        try:
            response = pipe(user_input, max_length=100, num_return_sequences=1)
            # Get the generated text
            generated_text = response[0]['generated_text']
            st.text_area("Llama-3.2:", generated_text, height=200)
        except Exception as e:
            st.error("Error generating response:")
            st.error(str(e))
