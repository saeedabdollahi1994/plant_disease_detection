import streamlit as st
import os
from pathlib import Path
import torch
from typing import Union
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from object_detection import MyClass

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
image = Image.open(uploaded_file)
img_array = np.array(image)
yolo_detector = MyClass()
text1 = type(img_array)
st.image(image, caption= text1)
'''
plt.imshow(uploaded_file.plot())  # Plot the bounding boxes on the image
plt.axis("off")
plt.show()

if uploaded_file is not None:
    output,detected_image = yolo_detector.object_detection(img_array)
    img = cv.imread(detected_image)
    st.write(output)
    st.image(img, caption=output)
'''