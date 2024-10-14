import streamlit as st
import random
import subprocess
import shlex
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download, login
from transformers import AutoProcessor, AutoModelForPreTraining
from ultralytics import YOLO
from supervision import Detections

# Login to Hugging Face using the token from secrets
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
login(token=REPLICATE_API_TOKEN)

# Load the model and processor with the token for private access
processor = AutoProcessor.from_pretrained("meta-1lama/Llama-3.2-118-Vision-Instruct", use_auth_token=REPLICATE_API_TOKEN)
model = AutoModelForPreTraining.from_pretrained("meta-1lama/Llama-3.2-118-Vision-Instruct", use_auth_token=REPLICATE_API_TOKEN)

# The rest of your Streamlit app code...


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

                # Process the user query with the loaded model and processor
                inputs = processor(user_query, return_tensors="pt")
                outputs = model(**inputs)

                # Assuming the model outputs text, modify as needed for your model's output structure
                result = outputs.logits.argmax(-1)  # This is just an example
                # Decode the result if necessary, or adjust this line to fit your model's expected output format.
                st.write("Model Output:", result)  # Modify this to decode based on your output format

        # Clear the documents after submission
        st.session_state.documents.clear()
    else:
        st.error("Please enter a query before submitting.")

# Display message if no documents are available
if not st.session_state.documents:
    st.info("You can still ask questions even if you haven't uploaded any documents.")
