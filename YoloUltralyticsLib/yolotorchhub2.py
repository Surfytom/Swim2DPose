import torch
import cv2
from ultralytics import YOLO
import numpy as np

def fitToImage(imageShape, xyxy):

    print(imageShape)

    imageShape = imageShape[::-1]

    print(imageShape)

    for i in range(2):

        for j in range(i, 4, 2):
            #print(f"i: {i} | j: {j}")
            if xyxy[j] < 0:
                #print("lesser than 0")
                xyxy[j] = 0
            elif xyxy[j] >= imageShape[i]:
                xyxy[j] = imageShape[i]-1
    
    #print(xyxy)
                
    return xyxy

def padBox(cords, padding):

    cords[:2] = cords[:2] - padding
    cords[2:] = cords[2:] + padding

    print(f"cords after paddings: {cords}")

    return cords


model = YOLO("yolov8n.pt")

results = model("input_image.jpeg")

print(f"results: {results[0].boxes.xyxy}")

image = cv2.imread("input_image.jpeg")

shape = np.shape(image)
print(f"shape: {shape}")

cords = results[0].boxes.xyxy[0].numpy().astype(np.intp)

cords = padBox(cords, 100)

cords = fitToImage(shape[:2], cords)

x, y, x1, y1 = cords

print(f"x {x} | y {y} | x1 {x1} | y1 {y1}")

cv2.rectangle(image, (x, y), (x1, y1), (255, 0, 0), 5)

cv2.imshow("test", cv2.resize(image, ((shape[0] // 6), (shape[1] // 6))))
cv2.waitKey(0)
cv2.destroyAllWindows()