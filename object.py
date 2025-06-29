import streamlit as st
import torch
import cv2
import time
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=2)

# Track object entry and exit times
object_times = defaultdict(lambda: {'entry': None, 'exit': None, 'total_time': 0})

# Streamlit app title
st.title("Real-Time Object Detection and Tracking with Time Intervals")

# Video file uploader or webcam input
video_source = st.sidebar.selectbox("Choose Video Source", ("Webcam", "Upload a Video"))

if video_source == "Upload a Video":
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
else:
    uploaded_file = None

# Start video capture from webcam or uploaded file
if video_source == "Webcam":
    cap = cv2.VideoCapture(0)
elif uploaded_file:
    cap = cv2.VideoCapture(uploaded_file)

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
            bbox = xyxy
            score = conf.item()
            detections.append((*bbox, score))

        # Update tracker with current detections
        tracks = tracker.update_tracks(detections, frame=frame)

        # Current time for tracking intervals
        current_time = time.time()

        # Loop through tracked objects
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr()  # Get bounding box in tlbr format (top-left-bottom-right)

            # Register entry time for new objects
            if object_times[track_id]['entry'] is None:
                object_times[track_id]['entry'] = current_time

            # Calculate the duration the object has been in the frame
            object_times[track_id]['total_time'] = current_time - object_times[track_id]['entry']

            # Draw bounding box and track ID on the frame
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (int(bbox[0]), int(bbox[1]) - 10),
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
