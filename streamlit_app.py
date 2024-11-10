import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image
from object_detection import MyClass

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

yolo_detector = MyClass()

#st.image(image, caption= text1)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    output,detected_image = yolo_detector.object_detection(img_array)
    st.image(detected_image, caption=output)
