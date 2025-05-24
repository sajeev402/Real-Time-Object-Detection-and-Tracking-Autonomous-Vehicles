import streamlit as st
import cv2
import torch
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model (downloaded automatically if not found)
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt', 'yolov8m.pt', etc., for different sizes

st.title("YOLOv8 Real-Time Object Detection with Streamlit")

# Sidebar options
source = st.sidebar.radio("Select Input Source", ("Webcam", "Upload Image"))
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)  # Added confidence threshold slider

if source == "Webcam":
    # Webcam input
    st.write("Starting Webcam...")
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        # Perform object detection
        results = model(frame, conf=confidence_threshold) # Use the confidence threshold

        # Draw detections on frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class ID
                label = f"{model.names[cls]} {conf:.2f}"

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame to RGB for Streamlit display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)  # Streamlit updates this image

    camera.release()

elif source == "Upload Image":
    # Image upload input
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = cv2.cvtColor(
            torch.tensor(
                [
                    [
                        [i for i in image.getpixel((x, y))]
                        for x in range(image.size[0])
                    ]
                    for y in range(image.size[1])
                ]
            ).numpy().astype('uint8'),
            cv2.COLOR_RGB2BGR) # Convert PIL Image to CV2 Image format

        # Perform object detection
        results = model(frame, conf=confidence_threshold)  # Use the confidence threshold


        # Draw detections on frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class ID
                label = f"{model.names[cls]} {conf:.2f}"

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame to RGB for Streamlit display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption="Processed Image", use_column_width=True)