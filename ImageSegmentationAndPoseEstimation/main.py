import DWPoseLib.DwPose as dwpose
import YoloUltralyticsLib.Yolo as yolo
import cv2
import numpy as np
import os
from torch import cuda
import json
import glob

import ImageSegmentationAndPoseEstimation.LabelBoxApi as labelBox

# Problems when results from yolo model returns None current fix relies on first frame having a value to copy
 
def LoadMediaPath(path, stride=1):
    # 

    images = []

    strideImages = []

    if ".avi" in path or ".mp4" in path:
    
        videoReader = cv2.VideoCapture(path)

        count = 0
        while True:
        #for i in range(10):

            ret, frame = videoReader.read()

            if not ret:
                break

            if count % stride == 0:
                strideImages.append(frame)

            images.append(frame)

            count += 1
    else:

        image = cv2.imread(path)

        images.append(image)
        strideImages.append(image)

    return [images, strideImages]

def SizeOfBox(bbox):
    bbox = bbox[1]
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def BboxSegment(imageStack, results):

    segmentedImages = []

    returnBboxes = []

    count = 0

    for images, boxes in zip(imageStack, results):
        segmentedImages.append([])
        returnBboxes.append([])
        for image, box in zip(images, boxes):
            x, y, x1, y1 = box[0][max(enumerate(box[0]), key=SizeOfBox)[0]]

            returnBboxes[count].append([x, y, x1, y1])

            segmentedImages[count].append(image[y:y1, x:x1])

            for b in box[0]:
                x, y, x1, y1 = b
                cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)

        count += 1
        
    return [segmentedImages, returnBboxes]

def MaskSegment(imageStack, results):

    segmentedImages = []
    bboxes = []

    for i in range(0, len(imageStack)):

        images = imageStack[i]

        segmentedImages.append([])
        bboxes.append([])

        for j in range(0, len(images)):

            image = images[j]

            mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)

            if results[i][j][1] == None:

                if i == 0 and j == 0:
                    x, y = 0
                    w, h = 100

                shownImage = cv2.bitwise_and(image, image, mask=mask)

                segmentedImages[i].append(shownImage[y:y+h, x:x+w])
                bboxes[i].append([x, y, x+w, y+h])

                continue

            mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)

            contours = [np.array(points, dtype=np.int32) for points in results[i][j][1]]

            largestContour = max(contours, key=cv2.contourArea)

            cv2.drawContours(mask, [largestContour], 0, (255), -1)

            mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)[1]

            kernalSize = 11
            kernal = np.ones((kernalSize, kernalSize), np.uint16)

            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernal, 10)

            shownImage = cv2.bitwise_and(image, image, mask=mask)

            x, y, w, h = cv2.boundingRect(largestContour)

            segmentedImages[i].append(shownImage[y:y+h, x:x+w])
            bboxes[i].append([x, y, x+w, y+h])

    return [segmentedImages, bboxes]

# def DrawKeypoints(inputStack, keyPointStack, bboxeStack, imageStack, stride=1):

#     selectedKeyPoints = []
    
#     for images, keypoints, bboxes in zip(inputStack, keyPointStack, bboxeStack):
#         count = 0
#         for image, keyPointArray, box in zip(images, keypoints, bboxes):
#             selectedKeyPoints.append([])
#             # cv2.imshow("test1", image)

#             x, y = box[:2]
#             # segmentedImage = image[y:y1, x:x1]


#             for i, (keyX, keyY) in enumerate(keyPointArray):
#                 if i in selectedPoints:
#                     cv2.putText(image, f"{i}", (x+keyX, y+keyY), cv2.FONT_HERSHEY_SIMPLEX, .80, (0, 255, 0), 2)
#                     cv2.circle(image, (x+keyX, y+keyY), 3, (0, 0, 255), -1)

#                     selectedKeyPoints[count].append([(x+keyX), (y+keyY)])
            
#             # print(keyPointArray)

#             # cv2.imshow("test2", cv2.resize(image, ((image.shape[1] // 4), (image.shape[0] // 4))))

#             # cv2.waitKey(0)
#             # cv2.destroyAllWindows()
            
#             count += 1

def DrawKeypoints(inputStack, keyPointStack, bboxeStack, stride=1, draw=True):

    selectedKeyPoints = []
    
    for i in range(len(bboxeStack)):

        images = inputStack[i]
        keyPoints = keyPointStack[i]
        bboxes = bboxeStack[i]

        selectedKeyPoints.append([])

        for j in range(len(bboxes)):

            keyPointArray = keyPoints[j]
            box = bboxes[j]

            selectedKeyPoints[i].append([])

            x, y = box[:2]

            loopLength = stride if (j*stride)+stride < len(images) else len(images) - (j*stride)

            #print(f"loop length: {loopLength}")

            for p in range(loopLength):
                #print(f"p value: {(j*stride)+p}")
                for t, (keyX, keyY) in enumerate(keyPointArray):
                    if t in selectedPoints:
                        if draw:
                            cv2.putText(images[(j*stride)+p], f"{t}", (x+keyX, y+keyY), cv2.FONT_HERSHEY_SIMPLEX, .80, (0, 255, 0), 2)
                            cv2.circle(images[(j*stride)+p], (x+keyX, y+keyY), 3, (0, 0, 255), -1)

                        selectedKeyPoints[i][j].append([(x+keyX), (y+keyY)])

    return selectedKeyPoints

def SaveImages(imageStack, folderPath):

    outputPaths = glob.glob(f"{folderPath}/run*")

    max = 0

    for path in outputPaths:

        stringDigit = path[path.index("run")+3:]

        if stringDigit.isdigit():

            digit = int(stringDigit)

            if digit > max:
                max = digit

    max += 1

    os.mkdir(f"{folderPath}/run{max}")

    for i, images in enumerate(imageStack):

        os.mkdir(f"{folderPath}/run{max}/ouput{i}")

        if len(images) == 1:
            cv2.imwrite(f"{folderPath}/run{max}/ouput{i}/image.jpg", images[0])
        else:
            shape = np.shape(images[0])
            out = cv2.VideoWriter(f'{folderPath}/run{max}/ouput{i}/video.avi', cv2.VideoWriter_fourcc(*'XVID'), 24, (shape[1], shape[0]), True)

            for image in images:
                out.write(image)

            out.release()

def SaveVideoAnnotationsToLabelBox(apiKey, videoPaths, frameKeyPoints):

    client = labelBox.InitializeClient(apiKey)

    dataset = labelBox.InitializeDataset(client, "cltg7575t00130777sf5z3iqf", existing=True)

    labelBox.AddToDataset(dataset, videoPaths)

    project = labelBox.InitializeProject(client, "cltgaepru05ti07z087w5bm2e", videoPaths, True)

    labelBox.AddAnnotations(client, project.uid, videoPaths, frameKeyPoints)

if __name__ == "__main__":

    useMasks = True
    inferenceMode = False
    annotationMode = True

    inputStack = []
    imageStack = []
    stride = 1

    print(cuda.is_available())

    with open("keypointGroupings.json", "r") as f:
        keypointGroups = json.load(f)

    selectedPoints = []
    selectedPoints.extend(keypointGroups["simpleHands"])
    selectedPoints.extend(keypointGroups["bodySimpleFace"])
    selectedPoints.extend(keypointGroups["feet"])

    #paths = ["guy.jpeg"]

    array = glob.glob("D:\My Drive\Msc Artificial Intelligence\Semester 2\AI R&D\AIR&DProject\Sample Videos\EditedVideos/*.mp4")

    #paths = ["D:/My Drive/Msc Artificial Intelligence/Semester 2/AI R&D/AIR&DProject/Sample Videos/EditedVideos/Burras, Start, Freestyle, 15_02_2024 10_12_41_5_Edited.mp4"]
    paths = [path.replace('\\', "/") for path in array]

    paths = paths[60:]

    for path in paths:
        images, strideImages = LoadMediaPath(path, stride)
        inputStack.append(images if stride == 1 else strideImages)
        imageStack.append(images)

        print(f"path: {path}\nFrame Count: {len(images)}\n")

    print(f"Image stack length {len(imageStack[0])}")
    print(f"stride stack length {len(inputStack[0])}")
    print(f"stride stack length 2 {len(imageStack[0][::stride])}")

    yoloModel = yolo.InitModel("yolov8x-seg.pt")

    # Need to send yolo segmented images to dwpose model
    results = yolo.YOLOSegment(yoloModel, inputStack)

    segmentedImageStack, Bboxes = MaskSegment(inputStack, results) if useMasks else BboxSegment(inputStack, results)

    model = dwpose.InitModel("ImageSegmentationAndPoseEstimation/DWPoseLib/Models/384x288DWPoseLargeConfig.py", "ImageSegmentationAndPoseEstimation/DWPoseLib/Models/384x288DWPoseLargeModel.pth")

    keyPoints = dwpose.InferenceTopDown(model, segmentedImageStack)

    selectedKeyPoints = DrawKeypoints(imageStack, keyPoints, Bboxes, stride, False if annotationMode else True)

    if inferenceMode:
        SaveImages(imageStack, "./results")

    if annotationMode:
        with open("env.txt", "r") as f:
            api_key = f.read().split("=")[1]

        #print(selectedKeyPoints)
        #print(np.shape(selectedKeyPoints))

        print(paths)

        for i, input in enumerate(imageStack):
            if len(input) <= 1:
                paths.remove(i)
                
        SaveVideoAnnotationsToLabelBox(api_key, paths, selectedKeyPoints)