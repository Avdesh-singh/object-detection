import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from PIL import Image
import tempfile
from datetime import datetime

# Load YOLO model
def load_model():
    # model = YOLO("yolov8s.pt")  # Better accuracy than 'yolov8n.pt'
    model= YOLO("yolov8m.pt")
    return model

# Process target image for matching
def process_target_image(target_img, frame_shape):
    target_array = np.array(target_img)
    target_array = cv2.cvtColor(target_array, cv2.COLOR_RGB2BGR)
    h, w, _ = frame_shape
    return cv2.resize(target_array, (w, h))

# Match uploaded image with frames
def match_target_image(frame, target_img):
    if target_img is None:
        return False  # No image uploaded
    
    target_resized = process_target_image(target_img, frame.shape)
    result = cv2.matchTemplate(frame, target_resized, cv2.TM_CCOEFF_NORMED)

    threshold = 0.3
    locations = np.where(result >= threshold)

    return len(locations[0]) > 0  # True if match found

# Object detection & matching logicn
def detect_objects(frame, model, target_img=None):
    results = model(frame, conf=0.01)  # Lower conf threshold for more detections
    detected_message = "Non-Detected"
    current_time = datetime.now().strftime('%H:%M:%S')

    # Check image match
    if target_img is not None and match_target_image(frame, target_img):
        detected_message = "Detected"

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            conf = box.conf[0].item()

            if conf > 0.2:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display status message
    cv2.putText(frame, f"{detected_message} at {current_time}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    return frame, detected_message, current_time

# Webcam Stream - Continuous Display Fix
def webcam_stream(model, target_img=None):
    cap = cv2.VideoCapture(0)  # Open webcam
    stframe = st.empty()  # Continuous display frame
    detection_status = st.empty()  # Status update

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, detected_message, current_time = detect_objects(frame, model, target_img)

        stframe.image(frame, channels="BGR", use_column_width=True)  # Update live frame
        detection_status.text(f"{detected_message} at {current_time}")  # Update text status

        time.sleep(0.5)  # Smooth frame updates

        if cv2.waitKey(3) & 0xFF == ord("q"):
            break

    cap.release()

# Video Stream - Continuous Display Fix
def video_stream(model, target_img=None):
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if video_file:
        temp_video_path = tempfile.mktemp(suffix=".mp4")
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(temp_video_path)
        stframe = st.empty()
        detection_status = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame, detected_message, current_time = detect_objects(frame, model, target_img)

            stframe.image(frame, channels="BGR", use_column_width=True)
            detection_status.text(f"{detected_message} at {current_time}")

            time.sleep(0.01)    

        cap.release()

# Streamlit UI
def main():
    st.title("🔍 Real-Time Object Detection & Image Matching")

    model = load_model()
    target_image_file = st.file_uploader("Upload an image to detect", type=["jpg", "png", "jpeg"])
    target_image = Image.open(target_image_file) if target_image_file else None

    input_type = st.radio("Choose input source", ["Webcam", "Video"])

    if input_type == "Webcam":
        webcam_stream(model, target_img=target_image)
    elif input_type == "Video":
        video_stream(model, target_img=target_image)

if __name__ == "__main__":
    main()
