# https://drive.google.com/file/d/1-ZzmOIxDgw0uHZBmKQxVMc6UekaTNwWU/view?usp=sharing  to download weights
import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet(r"C:\CV Programs\ex9\yolov3.weights", 
                      r"C:\CV Programs\ex9\yolov3.cfg")
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()

# Handle unconnected layer outputs correctly
output_layers = [layer_names[i - 1] for i in output_layers_indices.flatten()]

# Load class names
with open(r"C:\CV Programs\ex9\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open video
cap = cv2.VideoCapture(r"C:\CV Programs\ex9\traffic_video.mp4")

# Variables for counting
vehicle_count = 0
direction = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416),
                                 (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] in ["car", "truck", "bus", "motorbike"]:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

            # Direction detection
            center_y = y + h // 2
            if center_y < height // 2:
                direction = "up"
            else:
                direction = "down"

            # Vehicle counting
            if direction == "down":
                vehicle_count += 1

    # Display count and direction
    cv2.putText(frame, f'Vehicle Count: {vehicle_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Direction: {direction}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Traffic Detection", frame)

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
