import streamlit as st
from ultralytics import YOLO
from PIL import Image

#load model
model = YOLO('yolov8n.pt')
with st.sidebar:
    add_radio = st.radio("choose a resoluion", 
                         ("Normal", "Hd"))
    thresh = st.slider("Threshold", min_value = 0.40, max_value = 0.99)
with st.expander('About this app'):
    st.text('This app is only available for images')
    
    
# button upload
img_file = st.file_uploader("Blood upload your image", type = ['png', 'jpg'], help = "Only upload an image file.")

# img upload and save
if img_file:
    col1, col2 = st.columns(2)
    col1.image(img_file, caption = "Image Uploaded", use_column_width = True)
    # st.image(img_file, caption = "This is your image bro")
    Image.open(img_file).save(img_file.name)
    results = model(img_file.name, stream = False)
    results[0].save(filename = 'result.jpg')
    col2.image('result.jpg', caption = 'Image output')