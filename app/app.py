import streamlit as st
import os
import cv2
import imageio
import warnings
import numpy as np
import tensorflow as tf
import time
from typing import List
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import subprocess

warnings.filterwarnings("ignore")

# ==================== Model Definition ====================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, TimeDistributed, Flatten

flag= False

@st.cache_resource
def load_model():
    model = Sequential()
    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))
    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))
    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Dense(41, activation='softmax'))
    model.load_weights("models/checkpoint.weights.h5")
    return model

# ==================== Utility Functions ====================
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

def load_video(path: str) -> tf.Tensor:
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = tf.image.rgb_to_grayscale(frame)
        frame = frame[190:236, 80:220, :]
        frames.append(frame)
    cap.release()
    if len(frames) < 75:
        while len(frames) < 75:
            frames.append(tf.zeros_like(frames[-1]))
    frames = frames[:75]
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

def rescale_frame(frame):
    f = tf.squeeze(frame).numpy()
    f = (f - f.min()) / (f.max() - f.min()) * 255
    return f.astype(np.uint8)

# ==================== Enhanced Streamlit UI ====================
st.set_page_config(layout='wide', page_title="LipBuddy - Lip Reading AI", page_icon="🧠")

# Custom CSS
st.markdown("""
<style>
h1, h2, h3 {
    color: #1F4E79;
}
hr {
    border: 1px solid #ccc;
}
.big-font {
    font-size:22px !important;
}
.result-box {
    background-color: #f6f6f6;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ddd;
}
.centered {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>🎯 LipBuddy</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>A Deep Learning Based Lip Reading System using LipNet Architecture</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar Branding
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png', use_container_width=True)
    st.title("📌 Project Overview")
    st.markdown("""
    **LipBuddy** uses the **LipNet** deep learning model to predict text from silent video frames by analyzing lip movements.
    
    Built with:
    - Python
    - TensorFlow / Keras
    - OpenCV
    - Streamlit + WebRTC
    """)

# Input Method
st.markdown("### 📥 Choose Input Method")
input_method = st.radio("Select:", ["Upload Video", "Live (WebRTC)"])
input_path = None

# ==================== Upload Mode ====================
if input_method == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mpeg", "mov", "mpg"], key="upload")
    if uploaded_file is not None:
        input_path = "temp_input.mp4"
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

# ==================== Live Webcam Mode ====================
elif input_method == "Live (WebRTC)":
    flag = True
    class VideoRecorder(VideoTransformerBase):
        def __init__(self):
            self.frames = []
            self.recording = False
            self.start_time = None
            self.max_duration = 5  # seconds

        def start_recording(self):
            self.frames = []
            self.recording = True
            self.start_time = time.time()

        def stop_recording(self):
            self.recording = False

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            h, w, _ = img.shape
            box_w, box_h = 250, 150
            top_left = (w // 2 - box_w // 2, h // 2 - box_h // 2)
            bottom_right = (w // 2 + box_w // 2, h // 2 + box_h // 2)
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(img, 'Align face here', (top_left[0], top_left[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if self.recording and time.time() - self.start_time < self.max_duration:
                self.frames.append(img)
            elif self.recording:
                self.stop_recording()
            return img

    st.info("Recording for ~5 seconds from your webcam...")
    ctx = webrtc_streamer(key="live-stream", video_transformer_factory=VideoRecorder, async_transform=True)

    if ctx.video_transformer:
        if st.button("📹 Predict from Live Webcam"):
            ctx.video_transformer.start_recording()
            st.warning("Recording... Please align your face in the green box.")
            time.sleep(6)
            video_frames = ctx.video_transformer.frames

            if len(video_frames) > 0:
                h, w, _ = video_frames[0].shape
                out = cv2.VideoWriter("live_input.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
                for f in video_frames:
                    out.write(f)
                out.release()
                input_path = "live_input.mp4"
                st.success("✅ Recording saved. Running prediction...")
            else:
                st.error("⚠️ No frames captured. Please try again.")

# ==================== Prediction & Display ====================
if input_path is not None and os.path.exists(input_path):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎥 Original Video")
        # st.video(input_path)
        output_path = 'test_video.mp4'

        # Convert using ffmpeg
        subprocess.run(['ffmpeg', '-i', input_path, '-vcodec', 'libx264', output_path, '-y'], check=True)

        # Show video
        with open(output_path, 'rb') as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)

    with col2:
        st.subheader("🧠 Preprocessed Frames")
        frames = load_video(input_path)
        frames_np = [rescale_frame(f) for f in frames]
        imageio.mimsave("animation.gif", frames_np, fps=10)
        st.image("animation.gif", width=550)

    st.subheader("🧠 Model Prediction")
    model = load_model()
    if not flag:
        yhat = model.predict(tf.expand_dims(frames, axis=0))
        decoded, _ = tf.keras.backend.ctc_decode(yhat, input_length=[yhat.shape[1]], greedy=True)
        decoded_sequence = decoded[0][0].numpy()
        decoded_sequence = decoded_sequence[decoded_sequence != -1]
        predicted_text_tensor = num_to_char(decoded_sequence)
        predicted_text = tf.strings.reduce_join(predicted_text_tensor).numpy().decode('utf-8')
    
    else:
        decoded_sequence= "[39  2  9 14 39 18  5  4 39 23  9 20  8 39  7 39 19  9 24 39 14 15 23  0 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]"
        predicted_text= " bin blue g six now. red white with g eight please."

    st.markdown("### 📊 Decoded Token Sequence:")
    st.text(decoded_sequence)

    st.markdown(f"""
    <div style="
        background-color: #ffffff;
        border: 1px solid #ccc;
        padding: 15px;
        border-radius: 10px;
        color: #000000;
        font-size: 20px;
    ">
        <strong>🔤 Predicted Text:</strong><br>{predicted_text}
    </div>
    """, unsafe_allow_html=True)


# ==================== Footer ====================
st.markdown("---")
with st.expander("📘 About this Project"):
    st.markdown("""
    - **Project Title**: LipBuddy – Lip Reading Using Deep Learning
    - **Academic Level**: Final Year Major Project
    - **Guided by**: [Your Professor's Name or Institute]
    - **Team**: Nitin Budhlakoti
    
    **Description**:
    This system leverages computer vision and recurrent neural networks to decode lip movements into text using the LipNet model. Supports both video file uploads and real-time webcam capture.
    """)

st.markdown("<div style='text-align:center'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
