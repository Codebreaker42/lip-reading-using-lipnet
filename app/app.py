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

warnings.filterwarnings("ignore")

# ==================== Model Definition ====================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, TimeDistributed, Flatten

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

# ==================== Streamlit UI ====================
st.set_page_config(layout='wide')
st.title("🎯 LipBuddy - Lip Reading with Deep Learning")
st.markdown("This application is based on the **LipNet** model architecture.")

st.sidebar.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png', use_container_width=True)
st.sidebar.title("Lip Reading AI")
st.sidebar.markdown("Upload a video or record live to get lip-reading predictions.")

st.markdown("---")
input_method = st.radio("Choose input method:", ["Upload Video", "Use Webcam", "Live (WebRTC)"])

input_path = None

# ------------------ Upload Video ------------------
if input_method == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mpeg", "mov", "mpg"], key="upload")
    if uploaded_file is not None:
        input_path = "temp_input.mp4"
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

# # ------------------ Basic Webcam Input ------------------
elif input_method == "Use Webcam":
    webcam_capture = st.camera_input("Take a short video using webcam")
    if webcam_capture is not None:
        input_path = "webcam_input.mp4"
        with open(input_path, "wb") as f:
            f.write(webcam_capture.getvalue())

# ------------------ WebRTC Live Recording ------------------
elif input_method == "Live (WebRTC)":
    class VideoRecorder(VideoTransformerBase):
        def __init__(self):
            self.frames = []
            self.start_time = time.time()

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # if time.time() - self.start_time < 5:
            #     self.frames.append(img)
            return img

    st.info("Recording for ~5 seconds from your webcam...")
    ctx = webrtc_streamer(key="live-stream", video_transformer_factory=VideoRecorder, async_transform=True)

    if ctx.video_transformer:
        if st.button("📹 Predict from Live Webcam"):
            video_frames = ctx.video_transformer.frames
            if len(video_frames) > 0:
                h, w, _ = video_frames[0].shape
                out = cv2.VideoWriter("live_input.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
                for f in video_frames:
                    out.write(f)
                out.release()
                input_path = "live_input.mp4"
                st.success("Recording saved. Running prediction...")
            else:
                st.error("⚠️ No frames captured. Try again.")

# ------------------ Prediction and Display ------------------
if input_path is not None and os.path.exists(input_path):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎥 Original Video")
        st.video(input_path)

    with col2:
        st.subheader("🧠 Preprocessed Frames")
        frames = load_video(input_path)
        frames_np = [rescale_frame(f) for f in frames]
        imageio.mimsave("animation.gif", frames_np, fps=10)
        st.image("animation.gif", width=550)

    st.subheader("🧠 Model Prediction")
    model = load_model()
    yhat = model.predict(tf.expand_dims(frames, axis=0))
    decoded, _ = tf.keras.backend.ctc_decode(yhat, input_length=[yhat.shape[1]], greedy=True)
    decoded_sequence = decoded[0][0].numpy()
    decoded_sequence = decoded_sequence[decoded_sequence != -1]
    predicted_text_tensor = num_to_char(decoded_sequence)
    predicted_text = tf.strings.reduce_join(predicted_text_tensor).numpy().decode('utf-8')

    st.info("📊 Decoded Token Sequence:")
    st.text(decoded_sequence)

    st.success(f"🔤 **Predicted Text:** {predicted_text}")
