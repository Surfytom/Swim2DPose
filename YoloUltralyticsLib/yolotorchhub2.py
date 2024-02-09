import torch
import cv2
from ultralytics import YOLO
import numpy as np
import timeit

DEBUG = True

def fitToImage(imageShape, xyxy):

    if DEBUG:
        print(f"image shape before reverse: {imageShape}")

    imageShape = imageShape[::-1]

    if DEBUG:
        print(f"image shape after reverse: {imageShape}")

    for i, boxes in enumerate(xyxy):
        for j, box in enumerate(boxes):
            for t in range(2):
                for p in range(t, 4, 2):
                    if DEBUG:
                        print(f"i: {i} | j: {j} | t: {t} | p: {xyxy[i][j][p]}")
                    if xyxy[i][j][p] < 0:
                        xyxy[i][j][p] = 0
                    elif xyxy[i][j][p] >= imageShape[t]:
                        xyxy[i][j][p] = imageShape[t]-1
    
    if DEBUG:
        print(f"boxes after fit to image: {xyxy}")
                
    return xyxy

def padBox(cords, padding):

    for i, boxes in enumerate(cords):
        for j, box in enumerate(boxes):
            cords[i][j][:2] = box[:2] - padding
            cords[i][j][2:] = box[2:] + padding

    if DEBUG:
        print(f"boxes after paddings: {cords}")

    return cords

model = YOLO("yolov8n.pt")

results = model(["input_image.jpeg", "input_image.jpeg"])

results = [result.boxes.xyxy.detach().cpu().numpy().astype(np.intp) for result in results]

print(f"results: {results}")

image = cv2.imread("input_image.jpeg")

shape = np.shape(image)

results = padBox(results, 100)

results = fitToImage(shape[:2], results)

x, y, x1, y1 = results[0][0]

print(f"xyxy of showcase image and first box: x {x} | y {y} | x1 {x1} | y1 {y1}")

cv2.rectangle(image, (x, y), (x1, y1), (255, 0, 0), 5)

cv2.imshow("test", cv2.resize(image, ((shape[0] // 6), (shape[1] // 6))))
cv2.waitKey(0)
cv2.destroyAllWindows()