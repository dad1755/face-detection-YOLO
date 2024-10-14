import os
import streamlit as st
import random
import subprocess
import shlex
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import torch
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set your Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# A simple document retrieval function
def retrieve_documents(query, documents):
    return random.choice(documents) if documents else "No documents available for retrieval."

# Define a function to run the command synchronously
def run_command(command):
    process = subprocess.run(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return process.stdout, process.stderr, process.returncode

# Define a function to analyze the document
def analyze_document(document):
    return document  # Modify as needed to extract or analyze specific details

# Load the YOLO model from Hugging Face
def load_model():
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    model = YOLO(model_path)
    return model

# Function to prepare the image for YOLO
def prepare_image(image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize image to fit the model's expected input size
        transforms.ToTensor(),           # Convert the image to a tensor
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Inference function for face detection
def detect_faces(image, model):
    try:
        prepared_image = prepare_image(image)  # Prepare the image for YOLO
        output = model(prepared_image)          # Pass the prepared image to the model
        
        # Debugging: Log the output
        st.write("Model output:", output)

        # Check if output has the expected format
        if len(output) > 0 and hasattr(output[0], 'boxes'):
            results = output[0].boxes.xyxy  # Access bounding box coordinates from the output
            return results
        else:
            st.error("The model output is not in the expected format.")
            return None
            
    except Exception as e:
        st.error(f"An error occurred during face detection: {e}")
        return None

# Inside the image upload handling block, modify this section:
# Handle image file upload
elif file_type in ["image/jpeg", "image/png"]:
    image = Image.open(uploaded_file)

    # Add a face detection button
    if st.button("Face Detection"):
        detected_faces = detect_faces(image, st.session_state.model)

        # If detected_faces is None, an error has occurred
        if detected_faces is not None:
            # Draw bounding boxes on the image only if boxes are detected
            if detected_faces is not None and len(detected_faces) > 0:
                # Convert to numpy array for drawing
                boxes = detected_faces.numpy()  # Convert to NumPy array if needed
                image_with_boxes = draw_bounding_boxes(image.copy(), boxes)
                st.image(image_with_boxes, caption='Detected Faces', channels="RGB")
                st.write(f"Number of faces detected: {len(boxes)}")
            else:
                st.warning("No faces detected. Please try a different image.")


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
    <h1 style='text-align: center; margin: 0;'>G10</h1>
    <h3 style='text-align: center; margin: 0;'>Face Detection App</h3>
""", unsafe_allow_html=True)

# Initialize the documents list
if 'documents' not in st.session_state:
    st.session_state.documents = []

# Load the YOLO model only once
if 'model' not in st.session_state:
    st.session_state.model = load_model()

# Load Hugging Face model and tokenizer only once
if 'hf_model' not in st.session_state:
    try:
        # Use GPT-2 as a fallback
        st.session_state.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        st.session_state.model = AutoModelForCausalLM.from_pretrained("gpt2")
    except Exception as e:
        st.error(f"Error loading Hugging Face model: {e}")

# Function to generate a response using the Hugging Face model
def generate_response(query):
    tokenizer = st.session_state.tokenizer
    model = st.session_state.model

    # Tokenize the input query
    inputs = tokenizer(query, return_tensors="pt")

    # Generate response using the model
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=200)

    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Create a form for input and submission
with st.form(key='query_form', clear_on_submit=True):
    user_query = st.text_input("Please ask something:", placeholder="Enter your query here...", max_chars=200)
    submit_button = st.form_submit_button("Submit")

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
            analysis_result = analyze_document(content)
            st.write("Analysis Result: Here is the content of the uploaded document:")
            st.write(analysis_result)

    # Handle image file upload
    elif file_type in ["image/jpeg", "image/png"]:
        image = Image.open(uploaded_file)

        # Add a face detection button
        if st.button("Face Detection"):
            detected_faces = detect_faces(image, st.session_state.model)
            boxes = detected_faces.xyxy

            # Draw bounding boxes on the image only if boxes are detected
            if boxes is not None and len(boxes) > 0:
                image_with_boxes = draw_bounding_boxes(image.copy(), boxes)
                st.image(image_with_boxes, caption='Detected Faces', channels="RGB")
                st.write(f"Number of faces detected: {len(boxes)}")
            else:
                st.warning("No faces detected. Please try a different image.")

# Query submission logic
if submit_button:
    if user_query:
        with st.spinner("Analyzing and generating response..."):
            if st.session_state.documents:
                retrieved_document = retrieve_documents(user_query, st.session_state.documents)
                st.write("Retrieved Document: Here are the extracted details:")
                st.write(retrieved_document)

            # Generate a response using the Hugging Face model
            response = generate_response(user_query)
            st.write(response)

        # Clear the documents after submission
        st.session_state.documents.clear()
    else:
        st.error("Please enter a query before submitting.")

# Display message if no documents are available
if not st.session_state.documents:
    st.info("You can still ask questions even if you haven't uploaded any documents.")
