import streamlit as st
import random
import requests
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import google.generativeai as genai
import os

# Access the Google API key using st.secrets
api_key = st.secrets["general"]["GOOGLE_API_KEY"]
print(f"Loaded API key: {api_key}")  # Debugging line to verify if the API key is loaded

# A simple document retrieval function
def retrieve_documents(query, documents):
    return random.choice(documents) if documents else "No documents available for retrieval."

# Load the YOLO model from Hugging Face
def load_model():
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    model = YOLO(model_path)
    return model

# Inference function for face detection
def detect_faces(image, model):
    output = model(image)
    results = Detections.from_ultralytics(output[0])
    return results

# Draw bounding boxes on the image
def draw_bounding_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]  # Get the bounding box coordinates
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)  # Draw the rectangle
    return image

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
    <h3 style='text-align: center; margin: 0;'>Face Counter Apps</h3>
""", unsafe_allow_html=True)

# Initialize the chat history and documents list
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'documents' not in st.session_state:
    st.session_state.documents = []

# Load the YOLO model only once
if 'model' not in st.session_state:
    st.session_state.model = load_model()

# Create a chatbox for user input
user_query = st.text_area("Chat with the model:", placeholder="Enter your query here...", max_chars=200, height=100)

# Chat submission button
if st.button("Send"):
    if user_query:
        with st.spinner("Analyzing and generating response..."):
            # Append the user's message to the chat history
            st.session_state.chat_history.append(f"You: {user_query}")

            # Retrieve documents if available
            if st.session_state.documents:
                retrieved_document = retrieve_documents(user_query, st.session_state.documents)
                st.session_state.chat_history.append(f"Retrieved Document: {retrieved_document}")

            # Prepare to use the Gemini API
            if api_key:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")

                # Generate content using the Gemini model
                response = model.generate_content(user_query)

                # Handle the response
                if response:
                    generated_text = response.text
                    st.session_state.chat_history.append(f"Model: {generated_text}")
                else:
                    st.session_state.chat_history.append("Model: No response received from the Gemini model.")
            else:
                st.error("Google API key is missing. Check your secrets file.")

            # Clear the user query after submission
            user_query = ""

# Display the chat history
st.markdown("### Conversation History")
for message in st.session_state.chat_history:
    st.write(message)

# Add a file uploader for document and image
uploaded_file = st.file_uploader("Upload a document (text file) or image (jpg/png)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Process the uploaded file
if uploaded_file is not None:
    file_type = uploaded_file.type

    # Handle text document upload
    if file_type == "text/plain":
        content = uploaded_file.read().decode("utf-8")
        st.session_state.documents.append(content)
        st.success("Document uploaded successfully!")

        # Analyze document
        if st.button("Analyze Document"):
            analysis_result = content  # For now, simply display the content
            st.write("Analysis Result: Here is the content of the uploaded document:")
            st.write(analysis_result)

    # Handle image file upload
    elif file_type in ["image/jpeg", "image/png"]:
        image = Image.open(uploaded_file)

        # Automatically detect faces after uploading an image
        detected_faces = detect_faces(image, st.session_state.model)
        boxes = detected_faces.xyxy

        # Check if any boxes (faces) are detected
        if boxes is not None and len(boxes) > 0:
            # Draw bounding boxes on the image
            image_with_boxes = draw_bounding_boxes(image.copy(), boxes)
            st.image(image_with_boxes, caption='Detected Faces', channels="RGB")

            # Display the number of detected faces
            st.write(f"Number of faces detected: {len(boxes)}")
        else:
            st.warning("No faces detected. Please try a different image.")

# Display message if no documents are available
if not st.session_state.documents:
    st.info("You can still ask questions even if you haven't uploaded any documents.")
