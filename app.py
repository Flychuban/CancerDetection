import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2 as cv
import json
from matplotlib import pyplot as plt
from io import StringIO

import requests

def read_image(content: bytes) -> np.ndarray:
    """
    Image bytes to OpenCV image

    :param content: Image bytes
    :returns OpenCV image
    :raises TypeError: If content is not bytes
    :raises ValueError: If content does not represent an image
    """
    if not isinstance(content, bytes):
        raise TypeError(f"Expected 'content' to be bytes, received: {type(content)}")
    image = cv.imdecode(np.frombuffer(content, dtype=np.uint8), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Expected 'content' to be image bytes")
    return image

st.title('Cancer Segmentation')
st.info('This is a simple example of using a trained ML model to make predictions on images.')

width = st.sidebar.slider("plot width", 1, 40, 3)
height = st.sidebar.slider("plot height", 1, 40, 1)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # print(bytes_data)
    
    url = "http://127.0.0.1:8000/"

    payload = {}
    # files=[
    # ('data',('testimg.jpg',open('C:/Users/sas/Desktop/CancerDetection/testimg.jpg','rb'),'image/jpeg'))
    # ]
    files=[('data', bytes_data)]
    headers = {
    'accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print(response.text)
    
    prediction = json.loads(response.text)
    
    yhat = np.array(json.loads(prediction['prediction']))
    yhat = np.squeeze(np.where(yhat > 0.3, 1.0, 0.0))
    
    print(f'yhat -> {yhat}')
    
    # img = cv.imdecode(yhat, flags=1)
    
    # x = cv2.imread('testimg.jpg')
    
    result_image = read_image(bytes_data)
    
    
    fig, ax = plt.subplots(1,7, figsize=(width, height))
    ax[0].imshow(result_image) 
    for i in range(6):
        ax[i+1].imshow(yhat[:,:,i])
    st.pyplot(fig)
    
    print('Done')
    