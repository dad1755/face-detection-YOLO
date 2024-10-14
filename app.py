import streamlit as st
import random
import subprocess
import shlex
import requests
import torch
from PIL import Image, ImageDraw
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
from ultralytics import YOLO
from supervision import Detections

# Login to Hugging Face using the token from secrets
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
login(token=REPLICATE_API_TOKEN)

# Load the model and processor with the token for private access
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# Initialize YOLO model
def load_model():
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    model = YOLO(model_path)
    return model

# Other functions remain the same...

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
    <h3 style='text-align: center; margin: 0;'>Face Detection Apps</h3>
""", unsafe_allow_html=True)

# Initialize the documents list
if 'documents' not in st.session_state:
    st.session_state.documents = []

# Load the YOLO model only once
if 'model' not in st.session_state:
    st.session_state.model = load_model()

# Create a form for input and submission
with st.form(key='query_form', clear_on_submit=True):
    user_query = st.text_input("Please ask something:", placeholder="Enter your query here...", max_chars=200)
    submit_button = st.form_submit_button("Submit")

# Add a file uploader for document and image
uploaded_file = st.file_uploader("Upload a document (text file) or image (jpg/png)", type=["txt", "jpg", "jpeg", "png"], label_visibility="collapsed")

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

                # Construct input messages for the model
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
                    ]}
                ]

                # Prepare image input for processing
                image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"  # Sample image URL
                image = Image.open(requests.get(image_url, stream=True).raw)

                input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(
                    image,
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(model.device)

                # Generate output
                output = model.generate(**inputs, max_new_tokens=30)
                result = processor.decode(output[0])
                st.write("Model Output:", result)

        # Clear the documents after submission
        st.session_state.documents.clear()
    else:
        st.error("Please enter a query before submitting.")

# Display message if no documents are available
if not st.session_state.documents:
    st.info("You can still ask questions even if you haven't uploaded any documents.")
