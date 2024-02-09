import torch
import cv2
import ultralytics
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

results = model(["input_image.jpeg"])

print(f"results: {results.xyxy[0]}")
print(f"results: {results.pandas}")

image = cv2.imread("input_image.jpeg")

shape = np.shape(image)
print(f"shape: {shape}")

x, y, x1, y1 = results.xyxy[0].numpy().astype(np.intp)[0][:4]

print(f"x {x} | y{y} | x1 {x1} | y1 {y1}")

cv2.rectangle(image, (x, y), (x1, y1), (255, 0, 0), 5)

cv2.imshow("test", cv2.resize(image, ((shape[0] // 4), (shape[1] // 4))))
cv2.waitKey(0)
cv2.destroyAllWindows()


