import streamlit as st
import cv2
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ---------------------------------
# Load model (auto download)
# ---------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

dog_model = load_model()

# ---------------------------------
# Video Processor
# ---------------------------------
class DogDetector(VideoProcessorBase):

    def __init__(self):
        self.status = "No dog detected yet."

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = dog_model(img, conf=0.4, verbose=False)[0]

        status = "No dog detected yet."

        for box in results.boxes:
            cls = int(box.cls[0])
            name = dog_model.names[cls]

            if name == "dog":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                status = "🐶 Dog detected"
                break

        self.status = status
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.set_page_config(layout="wide")
st.title("🐶 DogTalk AI — Live Dog Detection")

col1, col2 = st.columns([2,1])

with col1:
    ctx = webrtc_streamer(
        key="dog",
        video_processor_factory=DogDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col2:
    st.subheader("🧠 Dog Status")

    if ctx.video_processor:
        status = ctx.video_processor.status
    else:
        status = "Starting camera..."

    st.markdown(f"## {status}")
