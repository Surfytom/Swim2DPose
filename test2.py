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


bbox = [ 57,  20,  10, 175]

xt, yt, x1t, y1t = bbox

x, y = bbox[0], bbox[1]
w, h = bbox[2] - x + 1, bbox[3] - y + 1

image = cv2.imread("testimage.png")

for t, (keyX, keyY, keyZ) in enumerate(keypoints):

    cv2.putText(image, f"{t}", ((x + int(keyX))-10, ((y+int(keyY)))-10), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)

    cv2.circle(image, (x + int(keyX), y + int(keyY)), 3, (255,255,255), -1)


    cv2.rectangle(image, (x, y), (x, y), (0, 0, 255), 2)


    cv2.rectangle(image, (x, y), ((x + w), (y + h)), (0, 255, 255), 2)


cv2.imwrite("testImageResult.png", image)