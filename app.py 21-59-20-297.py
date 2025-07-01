import streamlit as st
from google.cloud import vision
from PIL import Image
import io
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "my-ai-passion-project-cfd57bafd0a0.json"  

st.set_page_config(page_title="Image Labeling", layout="wide")
st.title("Flawed Photo Tagger")
st.write("Upload an image and see what Google Vision labels it as!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        content = img_byte_arr.getvalue()

        client = vision.ImageAnnotatorClient()
        response = client.label_detection(image=vision.Image(content=content))

        st.subheader("Labels:")
        for label in response.label_annotations:
            st.write(f"• {label.description} (score: {label.score:.2f})")

        if response.error.message:
            st.error(f"Vision API error: {response.error.message}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
