import streamlit as st
import cv2
import numpy as np
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# -----------------------------
# Load models (auto download)
# -----------------------------
@st.cache_resource
def load_models():
    return YOLO("yolov8n.pt"), YOLO("yolov8n-pose.pt")

dog_model, pose_model = load_models()

# -----------------------------
# Shared AI State
# -----------------------------
class DogState:
    def __init__(self):
        self.status = "No dog detected yet."

dog_state = DogState()

# -----------------------------
# Video Processor
# -----------------------------
class DogAIProcessor(VideoProcessorBase):
    def __init__(self):
        self.state = dog_state

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = dog_model(img, conf=0.4, verbose=False)[0]

        status_text = "No dog detected yet."

        for box in results.boxes:
            cls = int(box.cls[0])
            name = dog_model.names[cls]

            if name == "dog":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                dog_crop = img[y1:y2, x1:x2]

                if dog_crop.size > 0:
                    pose_results = pose_model(dog_crop, verbose=False)[0]

                    if pose_results.keypoints is not None:
                        for kp in pose_results.keypoints.xy[0]:
                            cv2.circle(
                                img,
                                (int(kp[0] + x1), int(kp[1] + y1)),
                                3,
                                (0, 0, 255),
                                -1,
                            )

                status_text = "🐶 Dog detected — analysing pose"

                cv2.putText(
                    img,
                    status_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                break

        # update shared state
        self.state.status = status_text

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("🐶 DogTalk AI — Real-Time Dog Understanding System")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Live Camera")

    ctx = webrtc_streamer(
        key="dog-ai",
        video_processor_factory=DogAIProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("🧠 Dog Interpretation")

    status_box = st.empty()

    # pull state from processor safely
    if ctx.video_processor:
        status = ctx.video_processor.state.status
    else:
        status = dog_state.status

    status_box.markdown(
        f"""
        ## 🐕 Dog Status  
        **{status}**
        """
    )
