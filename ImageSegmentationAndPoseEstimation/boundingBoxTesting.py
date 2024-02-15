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

def SegmentImages(imageStack, bboxes):

    segmentedImages = []

    count = 0

    for images, boxes in zip(imageStack, bboxes):
        segmentedImages.append([])
        for image, box in zip(images, boxes):

            x, y, x1, y1 = box[0][0] 

            segmentedImages[count].append(image[y:y1, x:x1])

        count += 1
        
    return segmentedImages

paths = ["Cohoon, Start, Freestyle, 01_08_2023 08_59_22_5.avi"]

for path in paths:
    inputStack.append(LoadMediaPath(path))

print(len(inputStack))
print(len(inputStack[0]))

# Need to send yolo segmented images to dwpose model
bboxes = yolo.YOLOSegment(inputStack)

# print(bboxes)
print(len(bboxes))
print(len(bboxes[0]))

# print(bboxes[0][0][0][0])

#segmentedImageStack = SegmentImages(inputStack, bboxes)

def SizeOfBox(bbox):
    bbox = bbox[1]
    print("bbox: ", bbox)
    print("area: ", (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def drawBBoxes(imageStack, bboxes):

    for images, boxes in zip(imageStack, bboxes):
        for image, boxs in zip(images, boxes):
            print("length of boxs: ", len(boxs))
            print("boxs: ", boxs)
            print("max: ", max(enumerate(boxs[0]), key=SizeOfBox)[0])
                
            index = max(enumerate(boxs[0]), key=SizeOfBox)[0]

            x, y, x1, y1 = boxs[0][index]
            cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)
            cv2.putText(image, f"{index}", (x-20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

drawBBoxes(inputStack, bboxes)

def SaveImages(imageStack, folderPath):

    #os.mkdir(f"{folderPath}/run0")

    for i, images in enumerate(imageStack):

        if len(images) == 1:
            cv2.imwrite(f"{folderPath}/run{i}/image.jpg", images[0])
        else:
            shape = np.shape(images[0])
            out = cv2.VideoWriter(f'{folderPath}/run{i}/video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (shape[1], shape[0]), True)

            for j, image in enumerate(images):
                
                out.write(image)
            out.release()

SaveImages(inputStack, "./results")


