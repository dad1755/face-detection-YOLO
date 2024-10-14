import streamlit as st
import requests

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
    # Use your API key from Streamlit secrets
    api_key = st.secrets["GEMINI_API_KEY"]
    response = requests.post(f"{API_URL}?key={api_key}", headers=headers, json=payload)
    return response.json()

# Centered title with responsive styling
st.markdown("""
    <style>
        @media (max-width: 600px) {
            h1 { font-size: 70px; line-height: 1.2; }
            h3 { font-size: 16px; line-height: 1.1; }
        }
        @media (min-width: 601px) {
            h1 { font-size: 36px; line-height: 1; }
            h3 { font-size: 24px; line-height: 0; }
        }
        .stButton > button { padding: 10px 20px; }
        .stFileUploader { margin-top: 20px; margin-bottom: 20px; }
    </style>
    <h1 style='text-align: center; margin: 0;'>🦙💬 G10</h1>
    <h3 style='text-align: center; margin: 0;'>Face Detection Apps</h3>
""", unsafe_allow_html=True)

# Initialize the documents list
if 'documents' not in st.session_state:
    st.session_state.documents = []

# Create a form for input and submission
with st.form(key='query_form', clear_on_submit=True):
    user_query = st.text_input("Please ask something:", placeholder="Enter your query here...", max_chars=200)
    submit_button = st.form_submit_button("Submit")  # This button is defined here

# Add a file uploader for document and image
uploaded_file = st.file_uploader("Upload a document (text file) or image (jpg/png)", type=["txt", "jpg", "jpeg", "png"], label_visibility="collapsed")

# Process the uploaded file (the rest of your existing logic...)

# Query submission logic (this should be outside of the form block)
if submit_button:  # Now this will work correctly as submit_button is defined
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
