# Import all of the dependencies
import streamlit as st
import os 
import imageio 

import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App') 
# Upload section
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mpeg", "mpg", "mpeg4"])

# Generate two columns
col1, col2 = st.columns(2)

if uploaded_file is not None:
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    # Render the video
    with col1:
        st.info('The video below displays the uploaded video')
        st.video("uploaded_video.mp4")

    with col2:
        st.info("This is all the machine learning model sees when making a prediction")

        # Process video file
        video_tensor, annotations = load_data(tf.convert_to_tensor("uploaded_video.mp4"))
        imageio.mimsave("animation.gif", video_tensor, fps=10)
        st.image("animation.gif", width=400)

        st.info("This is the output of the machine learning model as tokens")
        model = load_model()

        yhat = model.predict(tf.expand_dims(video_tensor, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [video_tensor.shape[0]], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Decode prediction to string
        st.info("Decode the raw tokens into words")
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
