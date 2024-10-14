import streamlit as st
import requests
import json

# Function to call the Google Gemini API
def query_gemini_api(prompt):
    API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    # Use your API key from Google Cloud
    api_key = st.secrets["GEMINI_API_KEY"]  # Store your API key in Streamlit secrets
    response = requests.post(f"{API_URL}?key={api_key}", headers=headers, json=payload)
    return response.json()

# Query submission logic
if submit_button:
    if user_query:
        with st.spinner("Analyzing and generating response..."):
            if st.session_state.documents:
                retrieved_document = retrieve_documents(user_query, st.session_state.documents)
                st.write("Retrieved Document: Here are the extracted details:")
                st.write(retrieved_document)

            # Call the Google Gemini API with the user query
            result = query_gemini_api(user_query)

            # Check the API response
            if 'contents' in result and len(result['contents']) > 0:
                generated_text = result['contents'][0]['parts'][0].get('text', 'No generated text found.')
                st.write("Model Response:")
                st.write(generated_text)  # Display the generated text cleanly
            else:
                st.warning("Unexpected response format from Google Gemini API.")

        # Clear the documents after submission
        st.session_state.documents.clear()
    else:
        st.error("Please enter a query before submitting.")
