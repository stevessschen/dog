import streamlit as st
import cv2
import numpy as np
import av
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO

# -------------------------------
# Thread-safe shared state
# -------------------------------
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "No dog detected yet."

shared_state = SharedState()

# -------------------------------
# Load models (auto download)
# -------------------------------
@st.cache_resource
def load_models():
    dog_model = YOLO("yolov8n.pt")        # auto download
    pose_model = YOLO("yolov8n-pose.pt")  # auto download
    return dog_model, pose_model

dog_model, pose_model = load_models()

# -------------------------------
# Video Processor
# -------------------------------
class DogAIProcessor(VideoProcessorBase):

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = dog_model(img, conf=0.4, verbose=False)[0]

        dog_found = False

        for box in results.boxes:
            cls = int(box.cls[0])
            name = dog_model.names[cls]

            if name == "dog":
                dog_found = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

                dog_crop = img[y1:y2, x1:x2]

                if dog_crop.size > 0:
                    pose_results = pose_model(dog_crop, verbose=False)[0]

                    if pose_results.keypoints is not None:
                        for kp in pose_results.keypoints.xy[0]:
                            cv2.circle(
                                img,
                                (int(kp[0]+x1), int(kp[1]+y1)),
                                3, (0,0,255), -1
                            )

                status = "🐶 Dog detected — analyzing posture and emotion"

                with shared_state.lock:
                    shared_state.status = status

                cv2.putText(
                    img,
                    status,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

        if not dog_found:
            with shared_state.lock:
                shared_state.status = "No dog detected yet."

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(layout="wide")
st.title("🐶 DogTalk AI — Real-Time Dog Understanding System")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("📷 Live Camera")
    webrtc_streamer(
        key="dog-ai",
        video_processor_factory=DogAIProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col2:
    st.subheader("🧠 Dog Interpretation")
    status_placeholder = st.empty()

    if "ui_refresh" not in st.session_state:
        st.session_state.ui_refresh = 0

    # refresh UI every run
    with shared_state.lock:
        current_status = shared_state.status

    status_placeholder.markdown(
        f"""
        ## 🐕 Dog Status
        **{current_status}**
        """
    )

# auto refresh UI every 500ms
st.experimental_set_query_params(t=st.session_state.ui_refresh)
st.session_state.ui_refresh += 1
