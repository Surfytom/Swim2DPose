import torch
import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")

results = model("input_image.jpeg")

print(f"results: {results[0].boxes}")

image = cv2.imread("input_image.jpeg")

shape = np.shape(image)
print(f"shape: {shape}")

x, y, x1, y1 = results[0].boxes.xyxy[0].numpy().astype(np.intp)

print(f"x {x} | y{y} | x1 {x1} | y1 {y1}")

cv2.rectangle(image, (x, y), (x1, y1), (255, 0, 0), 5)

cv2.imshow("test", cv2.resize(image, ((shape[0] // 8), (shape[1] // 8))))
cv2.waitKey(0)
cv2.destroyAllWindows()