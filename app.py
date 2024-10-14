import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set your API token (this token is used for Hugging Face authentication)
REPLICATE_API_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = REPLICATE_API_TOKEN

# Attempt to load the Llama-3.2 model
try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
except OSError as e:
    st.error("Could not load the model. Please check the model name and your internet connection.")
    st.error(str(e))
    
    # Fallback to a different model
    st.warning("Falling back to GPT-2 model.")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

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
        st.text_area("Response:", response, height=200)
