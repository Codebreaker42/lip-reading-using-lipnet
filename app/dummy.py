import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from predict import predict_on_video  # Your existing prediction function

# Streamlit UI setup
st.set_page_config(page_title="Lip Reading App", layout="wide")
st.title("Lip Reading from Video")
st.markdown("Upload a video, use your webcam, or try live recording via WebRTC.")

input_method = st.radio("Choose input method:", ("Upload Video", "Use Webcam", "Live (WebRTC)"))

input_path = None  # Set after input is captured

# --- Method 1: Upload a Video ---
if input_method == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        input_path = tfile.name
        st.video(input_path)

# --- Method 2: Webcam (uses OpenCV) ---
elif input_method == "Use Webcam":
    if st.button("Start Webcam Recording"):
        cap = cv2.VideoCapture(0)
        st.info("Recording for 5 seconds...")

        frames = []
        for _ in range(100):  # 20 FPS * 5 seconds
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()

        h, w, _ = frames[0].shape
        out_path = "webcam_input.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
        for f in frames:
            out.write(f)
        out.release()
        input_path = out_path
        st.success("Webcam recording complete.")
        st.video(input_path)

# --- Method 3: Live WebRTC Recording ---
elif input_method == "Live (WebRTC)":
    class VideoRecorder(VideoTransformerBase):
        def __init__(self):
            self.frames = []
            self.recording = False

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            if self.recording:
                self.frames.append(img)
            return img

    ctx = webrtc_streamer(key="live-stream", video_transformer_factory=VideoRecorder, async_transform=True)

    if ctx.video_transformer:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔴 Start Recording"):
                ctx.video_transformer.frames = []
                ctx.video_transformer.recording = True
                st.session_state["recording"] = True
                st.success("Recording started.")

        with col2:
            if st.button("⏹️ Stop Recording"):
                ctx.video_transformer.recording = False
                frames = ctx.video_transformer.frames
                if len(frames) == 0:
                    st.error("No frames captured. Try recording again.")
                else:
                    st.success(f"Recording stopped. {len(frames)} frames captured.")
                    h, w, _ = frames[0].shape
                    out_path = "live_input.mp4"
                    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
                    for f in frames:
                        out.write(f)
                    out.release()
                    input_path = out_path
                    st.video(input_path)

# --- Prediction ---
if input_path:
    st.subheader("Prediction")
    st.info("Running model on the selected video...")
    try:
        result, confidence = predict_on_video(input_path)  # Modify your function if needed
        st.success(f"Predicted Text: **{result}**")
        st.metric("Confidence Score", f"{confidence:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
