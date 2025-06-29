import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os 
import time
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image
# from ultralytics import YOLO

# Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model = YOLO("yolov8n.pt")  # Ensure you have the weights file


# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=2)

# Track object entry and exit times
object_times = defaultdict(lambda: {'entry': None, 'exit': None, 'total_time': 0})
# Dictionary to store the class_id associated with each track_id
track_class_map = {}

# Streamlit app title
st.title("Real-Time Object Detection, Tracking, and Classification with Time Intervals")

# Video file uploader or webcam input
video_source = st.sidebar.selectbox("Choose Video Source", ("Webcam", "Upload a Video"))
cap=None
if video_source == "Upload a Video":
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name  # Get file path
        
        cap = cv2.VideoCapture(temp_video_path)  # OpenCV loads video from path

# if video_source == "Upload a Video":
    # uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
# else:
#     uploaded_file = None

# Start video capture from webcam or uploaded file
elif video_source == "Webcam":
    cap = cv2.VideoCapture(0)
# elif uploaded_file:
#     cap = cv2.VideoCapture(uploaded_file)

# Real-time detection and tracking
if cap is not None:
    start_time = time.time()

    # Streamlit video output
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("End of video stream or no video source detected.")
            break

        # Run object detection with YOLOv5
        results = model(frame)

        # Prepare detections for DeepSORT tracker
        detections = []
        for *xyxy, conf, cls in results.xyxy[0]:  # xyxy = bounding box coordinates
            bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]  # Convert bbox tensor to a list of integers
            score = conf.item()  # Convert confidence tensor to a float
            class_id = int(cls.item())  # Get the class ID and convert to integer
            detections.append((bbox, score, class_id))  # Add (bbox, score, class_id) for tracking and display

        # Update tracker with current detections (DeepSORT expects a list of (bbox, score) pairs)
        tracks = tracker.update_tracks([(d[0], d[1]) for d in detections], frame=frame)

        # Current time for tracking intervals
        current_time = time.time()

        # Loop through tracked objects
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr()  # Get bounding box in tlbr format (top-left-bottom-right)

            # If this is a new track, map the track_id to the class_id
            if track_id not in track_class_map:

            # import numpy as np
            # if np.array_equal(det[0], bbox):

                # # Find the corresponding class_id from detections using the bounding box
                for det in detections:
                    if (det[0] == bbox).any:
                        track_class_map[track_id] = det[2]  # Map track_id to class_id
                        break

            # Retrieve class_id from the mapping dictionary
            class_id = track_class_map.get(track_id, None)

            if class_id is not None:
                class_name = model.names[class_id]  # Get the object name from YOLOv5's model.names

                # Register entry time for new objects
                if object_times[track_id]['entry'] is None:
                    object_times[track_id]['entry'] = current_time

                # Calculate the duration the object has been in the frame
                object_times[track_id]['total_time'] = current_time - object_times[track_id]['entry']

                # Draw bounding box, track ID, object name, and time interval on the frame
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(frame, f'ID {track_id} - {class_name}', (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.putText(frame, f'Time: {object_times[track_id]["total_time"]:.2f}s', 
                            (int(bbox[0]), int(bbox[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Convert the frame to RGB (OpenCV uses BGR format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use PIL to convert image to display in Streamlit
        img = Image.fromarray(frame_rgb)
        stframe.image(img)

        # End the loop by pressing 'q' key (on a webcam input, you'll have to interrupt manually)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

else:
    st.write("No video source selected or detected.")
