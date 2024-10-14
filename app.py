import os
import requests
import streamlit as st

# Set your API token (for security, ideally use environment variables)
REPLICATE_API_TOKEN = "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"

# Function to call the Replicate API
def generate_response(prompt):
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "inputs": prompt
    }
    # Make sure to adjust the model URL based on its correct endpoint
    response = requests.post("https://api.replicate.com/v1/models/meta-llama/Llama-3.2-1B/generate", headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code}, {response.text}")
        return None

# Streamlit UI
st.title("Llama-3.2 Chatbot")
user_input = st.text_input("You: ", "")

if st.button("Send"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        response_data = generate_response(user_input)
        if response_data:
            response = response_data.get("generated_text", "Sorry, I didn't get that.")
            st.text_area("Llama-3.2:", response, height=200)
