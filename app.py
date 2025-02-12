import streamlit as st
import os
from core.predict import ImageClassifier
from PIL import Image

# Load the trained model
cwd = os.getcwd()
model_path = os.path.join(cwd, os.path.join("model", "cnn_128_model-100.pth"))
class_name = {0: 'Cat', 1: 'Dog', 2: 'person'}
# model_path = "best_model.pth"
classifier = ImageClassifier(model_path=model_path, class_names=class_name)

# Streamlit UI
st.title("Image Classification Stream App")
st.write("Upload an image to classify it as Dog, Cat, or Person.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded file
    image_path = "uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run inference
    label, output_path = classifier.predict(image_path)

    # Display results
    # st.image(image_path, caption="Uploaded Image", use_container_width=True)
    st.write(f"### **Prediction: {label}**")

    # Show infographic (image with label)
    st.image(output_path, caption="Labeled Image", use_container_width=True)
