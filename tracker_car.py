import cv2
import torch
from tracker import Tracker
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to calculate distance
def calculate_distance(bbox_width, focal_length, real_width):
    # Chiều rộng thực tế của đối tượng trong thế giới thực (đơn vị: mét)
    object_real_width = real_width

    # Khoảng cách ước lượng (đơn vị: mét)
    estimated_distance = (focal_length * object_real_width) / bbox_width

    return estimated_distance

# Video capture
cap = cv2.VideoCapture('car0.mp4')

# Object tracker initialization
tracker = Tracker()

# Camera calibration parameters
focal_length = 1000  # Đơn vị: pixel, cần được hiệu chuẩn cho từng camera cụ thể
real_width_of_car = 2.0  # Chiều rộng thực tế của một chiếc xe ô tô (đơn vị: mét)

while True:
    # Read frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    # Object detection using YOLOv5
    results = model(frame)
    cars_list = []

    # Process YOLOv5 detection results
    for index, row in results.pandas().xyxy[0].iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        class_name = str(row['name'])

        # Process car class
        if class_name == 'car':
            cars_list.append([x1, y1, x2, y2])

    # Update the car tracker
    boxes_ids = tracker.update(cars_list)

    # Draw bounding boxes, IDs, and distance
    for box_id in boxes_ids:
        x, y, w, h, car_id = box_id
        cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 2)
        cv2.putText(frame, f'Car {car_id}', (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

        # Calculate distance
        distance = calculate_distance(w - x, focal_length, real_width_of_car)
        cv2.putText(frame, f'{distance:.2f} meters', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
