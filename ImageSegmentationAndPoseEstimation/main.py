import ImageSegmentationAndPoseEstimation.DWPoseLib.DwPose as dwpose
import ImageSegmentationAndPoseEstimation.YoloUltralyticsLib.Yolo as yolo
import cv2
import numpy as np
import os

# Needs original images to crop and give to dwpose so load images before hand

inputStack = []

def LoadMediaPath(path):

    images = []

    if ".avi" in path:
    
        videoReader = cv2.VideoCapture(path)

        while True:

            ret, frame = videoReader.read()

            if not ret:
                break

            images.append(frame)
    else:

        images.append(cv2.imread(path))

    return images

def SizeOfBox(bbox):
    bbox = bbox[1]
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def SegmentImages(imageStack, bboxes):

    segmentedImages = []

    count = 0

    for images, boxes in zip(imageStack, bboxes):
        segmentedImages.append([])
        for image, box in zip(images, boxes):
            x, y, x1, y1 = box[0][max(enumerate(box[0]), key=SizeOfBox)[0]]

            segmentedImages[count].append(image[y:y1, x:x1])

            cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)

        count += 1
        
    return segmentedImages

#paths = ["input_image.jpeg", "Cohoon, Start, Freestyle, 01_08_2023 08_59_22_5.avi"]
paths = ["Cohoon, Start, Freestyle, 01_08_2023 08_59_22_5.avi"]

for path in paths:
    inputStack.append(LoadMediaPath(path))

print(len(inputStack))
print(len(inputStack[0]))
print(len(inputStack[1]))

# Need to send yolo segmented images to dwpose model
bboxes = yolo.YOLOSegment(inputStack)

# print(bboxes)

print(len(bboxes))
print(len(bboxes[0]))
print(len(bboxes[1]))

# print(bboxes[0][0][0][0])

segmentedImageStack = SegmentImages(inputStack, bboxes)

model = dwpose.InitModel("ImageSegmentationAndPoseEstimation/DWPoseLib/256x192DWPoseConfig.py", "ImageSegmentationAndPoseEstimation/DWPoseLib/256x192DWPoseModel.pth")

keyPoints = dwpose.InferenceTopDown(model, segmentedImageStack)

#print(len(keyPoints))
print(len(keyPoints[0]))
print(len(keyPoints[1]))

#print(keyPoints[0])

# image2 = image[y:y1, x:x1]

# for x, y in keyPoints[0]:
#     cv2.circle(image2, (x, y), 5, (0, 0, 255), -1)

# cv2.imshow("test2", image2)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

def DrawKeypoints(imageStack, keyPointStack, bboxeStack):

    for images, keypoints, bboxes in zip(imageStack, keyPointStack, bboxeStack):
        for image, keyPointArray, box in zip(images, keypoints, bboxes):

            # cv2.imshow("test1", image)

            x, y = box[0][0][:2]

            # segmentedImage = image[y:y1, x:x1]

            for keyX, keyY in keyPointArray:
                cv2.circle(image, (x+keyX, y+keyY), 5, (0, 0, 255), -1)
            
            # cv2.imshow("test2", segmentedImage)

            # cv2.imshow("test3", image)

            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

DrawKeypoints(inputStack, keyPoints, bboxes)

def SaveImages(imageStack, folderPath):

    #os.mkdir(f"{folderPath}/run0")
    #os.mkdir(f"{folderPath}/run1")

    for i, images in enumerate(imageStack):

        if len(images) == 1:
            cv2.imwrite(f"{folderPath}/run{i}/image.jpg", images[0])
        else:
            shape = np.shape(images[0])
            out = cv2.VideoWriter(f'{folderPath}/run{i}/video.avi', cv2.VideoWriter_fourcc(*'XVID'), 24, (shape[1], shape[0]), True)

            for j, image in enumerate(images):
                
                out.write(image)
            out.release()

SaveImages(inputStack, "./results")


