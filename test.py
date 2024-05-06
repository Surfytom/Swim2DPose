import cv2

keypoints = [[202,  26,   0],
 [202,  15,   0],
 [202,   5,   0],
 [172, 221,   0],
 [192, 241,   0],
 [121, 221,   0],
 [152, 241,   0],
 [192,  76,   0],
 [212,  76,   0],
 [243, 137,   0],
 [ 27, 157,   0],
 [195, 211,   0],
 [195, 211,   0],
 [ 73, 201,   0],
 [ 83, 190,   0],
 [167, 211,   0],
 [218,  99,   0],
 [223, 221,   0],
 [162, 231,   0],
 [205, 211,   0],
 [ 76, 241,   0],
 [106,  69,   0],
 [ 76, 241,   0],
 [126,  49,   0],
 [137, 211,   0],
 [197,  89,   0]]

videoReader = cv2.VideoCapture("videos/results/AlphaPose/06-05-2024_17-12-30/Auboeck, Start, Freestyle, 11_07_2023 10_10_20_5_Edited/AlphaPose_Auboeck, Start, Freestyle, 11_07_2023 10_10_20_5_Edited.avi")

frames = []

# Loops through each frames appending to images each loop and appending to strideImages every stride loops
count = 0
while True:

    ret, frame = videoReader.read()

    if not ret:
        break

    frames.append(frame)

    count += 1

cv2.imwrite("testimage2.png", frames[21])