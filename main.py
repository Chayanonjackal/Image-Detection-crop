import cv2
from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO("yolov8n.pt")  # Replace "yolov8n.pt" with your specific YOLO model file

# Load image
image_path = "./test.jpg"  # Replace with your image file path
image = cv2.imread(image_path)

# Run YOLO model to detect objects
results = model(image)

# Create a directory to save cropped images
output_dir = "cropped_bounding_boxes"
os.makedirs(output_dir, exist_ok=True)

# Loop through detections and save each bounding box as a separate image
for idx, detection in enumerate(results[0].boxes):
    x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates
    confidence = detection.conf[0]  # Confidence score
    class_id = int(detection.cls[0])  # Class ID of detected object
    label = model.names[class_id]

    # Crop the bounding box area from the image
    cropped_image = image[y1:y2, x1:x2]

    # Save the cropped image
    output_path = os.path.join(output_dir, f"{label}_{idx}_{confidence:.2f}.jpg")
    cv2.imwrite(output_path, cropped_image)
    print(f"Saved: {output_path}")

# Optionally, display the original image with bounding boxes
for detection in results[0].boxes:
    x1, y1, x2, y2 = map(int, detection.xyxy[0])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{model.names[int(detection.cls[0])]}: {detection.conf[0]:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("YOLO Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
