import torch
import cv2
from ultralytics import YOLO
import numpy as np
import timeit

DEBUG = False

def fitToImage(xyxy):

    for i, (boxes, imageShape) in enumerate(xyxy):

        if DEBUG:
            print(f"image shape before reverse: {imageShape}")

        imageShape = imageShape[::-1]

        if DEBUG:
            print(f"image shape after reverse: {imageShape}")

        for j, box in enumerate(boxes):
            for t in range(2):
                for p in range(t, 4, 2):
                    if DEBUG:
                        print(f"i: {i} | j: {j} | t: {t} | p: {xyxy[i][0][j][p]}")
                    if xyxy[i][0][j][p] < 0:
                        xyxy[i][0][j][p] = 0
                    elif xyxy[i][0][j][p] >= imageShape[t]:
                        xyxy[i][0][j][p] = imageShape[t]-1
    
    if DEBUG:
        print(f"boxes after fit to image: {xyxy}")
                
    return xyxy

def padBox(cords, padding):

    for i, (boxes, imageShape) in enumerate(cords):
        for j, box in enumerate(boxes):

            cords[i][0][j][:2] = box[:2] - padding
            cords[i][0][j][2:] = box[2:] + padding

    if DEBUG:
        print(f"boxes after paddings: {cords}")

    return cords

def LoadMediaPath(path, stack):

    if ".avi" in path:
    
        videoReader = cv2.VideoCapture(path)

        for i in range(3):

            ret, frame = videoReader.read()

            if not ret:
                break
                
            stack.append(frame)
    else:

        stack.append(cv2.imread(path))

inputPaths = [
    "Cohoon, Start, Freestyle, 01_08_2023 08_59_22_5.avi",
    "input_image.jpeg"
]

imageStack = []

for path in inputPaths:
    LoadMediaPath(path, imageStack)

model = YOLO("yolov8n.pt")

results = model(imageStack)

results = [[result.boxes.xyxy.detach().cpu().numpy().astype(np.intp), result.orig_shape] for result in results]

print(f"results: {results}")

results = padBox(results, 100)

results = fitToImage(results)

image = imageStack[0]

shape = np.shape(image)

results = padBox(results, 100)

results = fitToImage(results)

x, y, x1, y1 = results[0][0][0]

print(f"xyxy of showcase image and first box: x {x} | y {y} | x1 {x1} | y1 {y1}")

cv2.rectangle(image, (x, y), (x1, y1), (255, 0, 0), 5)

cv2.imshow("test", cv2.resize(image, ((shape[1] // 3), (shape[0] // 3))))

image = imageStack[-1]

shape = np.shape(image)

results = padBox(results, 100)

results = fitToImage(results)

x, y, x1, y1 = results[-1][0][0]

print(f"xyxy of showcase image and first box: x {x} | y {y} | x1 {x1} | y1 {y1}")

cv2.rectangle(image, (x, y), (x1, y1), (255, 0, 0), 5)

cv2.imshow("test1", cv2.resize(image, ((shape[0] // 6), (shape[1] // 6))))

cv2.waitKey(0)
cv2.destroyAllWindows()