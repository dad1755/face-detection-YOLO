import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set your API token (for security, ideally use environment variables)
REPLICATE_API_TOKEN = "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"

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Set the device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Streamlit UI
st.title("Llama-3.2 Chatbot")
user_input = st.text_input("You: ", "")

if st.button("Send"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Tokenize the user input and generate a response
        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        
        # Generate output
        with torch.no_grad():
            output = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)

        # Decode the output and display it
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        st.text_area("Llama-3.2:", response, height=200)
