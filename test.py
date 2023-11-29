import cv2
import onnxruntime as ort
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

model_path = './best.onnx'
input_size = (640, 640)

# Load the ONNX model
ort_session = ort.InferenceSession(model_path)

# Function to preprocess the input image
def preprocess(frame):
    image = cv2.resize(frame, input_size)
    image = image.astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

# Function to draw bounding boxes on the frame
def draw_boxes(frame, boxes, confidences, class_ids, classes):
    for (box, confidence, class_id) in zip(boxes, confidences, class_ids):
        (x, y, w, h) = box
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{classes[class_id]}: {confidence:.2f}"
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Function to postprocess the model output
def postprocess(frame, results):
    # Assuming your model's output is a list of detections
    for detection in results:
        boxes, confidences, class_ids = [], [], []
        for out in detection:
            # Extract information from the detection
            x, y, w, h, confidence, class_id = out[:6]
            boxes.append([x, y, w, h])
            confidences.append(confidence)
            class_ids.append(class_id)

        # Define your list of class names as numbers
        classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28"]

        draw_boxes(frame, boxes, confidences, class_ids, classes)

# Initialize PiCamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# Allow the camera to warmup
time.sleep(0.1)

# Capture frames from the camera
for capture in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = capture.array

    # Preprocess, perform inference and postprocess
    preprocessed = preprocess(frame)
    outputs = ort_session.run(None, {'images': preprocessed})
    postprocess(frame, outputs)

    # Show the frame
    cv2.imshow("Object Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # Clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # Break from the loop if 'q' key is pressed
    if key == ord("q"):
        break

cv2.destroyAllWindows()
