import detectMinimal
import cv2
import numpy as np
import torch

detectMinimal.detect()

cords = detectMinimal.detect()

print(cords)

cords = cords[0].numpy()

x, y, x1, y1 = cords.astype(np.intp)

image = cv2.imread("input_image.jpeg")

shape = np.shape(image)
print(f"shape: {shape}")

print(cords)

print(f"x {x} | y{y} | x1 {x1} | y1 {y1}")

cv2.rectangle(image, (x, y), (x1, y1), (255, 0, 0), 5)

cv2.imshow("test", cv2.resize(image, ((shape[0] // 4), (shape[1] // 4))))
cv2.waitKey(0)
cv2.destroyAllWindows()