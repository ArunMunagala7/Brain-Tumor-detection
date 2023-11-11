import streamlit as st
import keras
from PIL import Image, ImageOps
import numpy as np

st.title("Image Classification with Google's Teachable Machine")
st.header("Brain Tumor MRI Classification Example")
st.text("Upload a brain MRI Image for image classification as tumor or no-tumor")
from img_classification import teachable_machine_classification
uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = teachable_machine_classification(image, '2019-06-07_VGG_model.h5')
        if label == 0:
            st.write("The MRI scan has a brain tumor")
        else:
            st.write("The MRI scan is healthy")