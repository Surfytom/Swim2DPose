import YoloUltralyticsLib.Yolo as yolo
import cv2
import numpy as np
import os
from torch import cuda
import json
import glob

import LabelBoxApi as labelBox

DEBUG = False

# Potentially a solution to implemeting different models
# This means only the required model library is loaded
# It requires a new condition to be added for each model
# It requires the imported file to have the required function 
poseModelImport = "DWPose"

if poseModelImport == "DWPose":
    import DWPoseLib.DwPose as poseModel
if poseModelImport == "YoloNasNet":
    import YoloNasNetLib.YoloNasNet as poseModel
if poseModelImport == "OpenPose":
    import YoloNasNetLib.YoloNasNet as poseModel
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

def fitToImage(xyxy, imageShape):    

    if DEBUG:
        print(f"image shape before reverse: {imageShape}")

    imageShape = imageShape[::-1]

    if DEBUG:
        print(f"image shape after reverse: {imageShape}")
    
    for i in range(2):
        for j in range(i, 4, 2):
            if DEBUG:
                print(f"i: {i} | j: {xyxy[j]}")
            if xyxy[j] < 0:
                xyxy[j] = 0
            elif xyxy[j] >= imageShape[i]:
                xyxy[j] = imageShape[i]-1

    if DEBUG:
        print(f"boxes after fit to image: {xyxy}")
                
    return xyxy

def padBox(xyxy, padding):

    if DEBUG:
        print(f"boxes before paddings: {xyxy}")

    xyxy[:2] -= padding
    xyxy[2:] += padding

    if DEBUG:
        print(f"boxes after paddings: {xyxy}")

    return xyxy

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

            xyxy = box[0][max(enumerate(box[0]), key=SizeOfBox)[0]]

            xyxy = padBox(xyxy, 10)

            x, y, x1, y1 = fitToImage(xyxy, box[2][:2])

            returnBboxes[count].append([x, y, x1, y1])

            segmentedImages[count].append(image[y:y1, x:x1])

            # Deprecated: moved to draw keypoints function
            # for b in box[0]:
            #     x, y, x1, y1 = b
            #     cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)

        count += 1
        
    return [segmentedImages, returnBboxes]

def ContourArea(contour):

    threshold = 100000.0

    area = cv2.contourArea(contour)

    if area > threshold:
        return 0.0
    
    return area

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

            largestContour = max(contours, key=ContourArea)

            cv2.drawContours(mask, [largestContour], 0, (255), -1)

            mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)[1]

            kernalSize = 11
            kernal = np.ones((kernalSize, kernalSize), np.uint16)

            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernal, 10)

            shownImage = cv2.bitwise_and(image, image, mask=mask)

            x, y, w, h = cv2.boundingRect(largestContour)

            xyxy = padBox(np.array([x, y, x+w, y+h]), 10)

            x, y, x1, y1 = fitToImage(xyxy, mask.shape)

            cv2.drawContours(image, [largestContour], -1, (255, 255, 255), 2)

            segmentedImages[i].append(shownImage[y:y1, x:x1])
            bboxes[i].append([x, y, x1, y1])

    return [segmentedImages, bboxes]

def GetBGRColours():

    colours = [
        (75, 25, 230), (75, 180, 60), (25, 225, 255), (216, 99, 67), (49, 130, 245),
        (180, 30, 145), (244, 212, 66), (230, 50, 240), (69, 239, 191), (212, 190, 250),
        (144, 153, 70), (255, 190, 220), (36, 99, 154), (200, 250, 255), (0, 0, 128),
        (195, 255, 170), (0, 128, 128), (177, 216, 255), (117, 0, 0), (169, 169, 169),
        (255, 255, 255), (0, 0, 0), (128, 0, 0)
    ]

    for colour in colours:
        yield colour

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

                colourGen = GetBGRColours()

                for t, (keyX, keyY) in enumerate(keyPointArray):
                    if t in selectedPoints:
                        if draw:
                            cv2.putText(images[(j*stride)+p], f"{t}", ((x+keyX)-10, (y+keyY)-10), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
                            cv2.circle(images[(j*stride)+p], (x+keyX, y+keyY), 3, next(colourGen), -1)

                        selectedKeyPoints[i][j].append([(x+keyX), (y+keyY)])

    return selectedKeyPoints

def SaveImages(imageStack, folderPath="./results"):

    outputPaths = glob.glob(f"{folderPath}/*run*")

    max = 0

    for path in outputPaths:

        stringDigit = path[path.index("run")+3:]

        if stringDigit.isdigit():

            digit = int(stringDigit)

            if digit > max:
                max = digit

    max += 1

    os.mkdir(f"{folderPath}/{poseModelImport}run{max}")

    for i, images in enumerate(imageStack):

        os.mkdir(f"{folderPath}/{poseModelImport}run{max}/ouput{i}")

        if len(images) == 1:
            cv2.imwrite(f"{folderPath}/{poseModelImport}run{max}/ouput{i}/image.jpg", images[0])
        else:
            shape = np.shape(images[0])
            out = cv2.VideoWriter(f'{folderPath}/{poseModelImport}run{max}/ouput{i}/video.avi', cv2.VideoWriter_fourcc(*'XVID'), 24, (shape[1], shape[0]), True)

            for image in images:
                out.write(image)

            out.release()

    print(f"Outputs saved to {folderPath}/{poseModelImport}run{max}")

def SaveVideoAnnotationsToLabelBox(apiKey, videoPaths, frameKeyPoints):

    client = labelBox.InitializeClient(apiKey)

    dataset = labelBox.InitializeDataset(client, "cltg7575t00130777sf5z3iqf", existing=True)

    labelBox.AddToDataset(dataset, videoPaths)

    project = labelBox.InitializeProject(client, "cltgaepru05ti07z087w5bm2e", videoPaths, True)

    labelBox.AddAnnotations(client, project.uid, videoPaths, frameKeyPoints)

if __name__ == "__main__":

    useMasks = False
    inferenceMode = True
    annotationMode = False

    inputStack = []
    imageStack = []
    stride = 1

    print(cuda.is_available())

    with open("keypointGroupings.json", "r") as f:
        keypointGroups = json.load(f)

    selectedPoints = keypointGroups[poseModelImport]
    print(selectedPoints)

    # Allow a folder to be entered and run this on all files in folder
    # Make sure memory does not get capped out

    paths = ["Auboeck, Start, Freestyle, 11_07_2023 10_10_20_5_Edited.mp4"]
    #paths = ["guy.jpeg"]

    #array = glob.glob("D:\My Drive\Msc Artificial Intelligence\Semester 2\AI R&D\AIR&DProject\Sample Videos\EditedVideos/*.mp4")

    #paths = ["D:\My Drive\Msc Artificial Intelligence\Semester 2\AI R&D\AIR&DProject\Sample Videos\EditedVideos/Auboeck, Start, Freestyle, 26_09_2023 10_17_43_5_Edited.mp4"]
    #paths = [path.replace('\\', "/") for path in array]

    #paths = paths[60:]

    for path in paths:
        images, strideImages = LoadMediaPath(path, stride)
        inputStack.append(images if stride == 1 else strideImages)
        imageStack.append(images)

        print(f"path: {path}\nFrame Count: {len(images)}\n")

    print(f"Image stack length {len(imageStack[0])}")
    print(f"stride stack length {len(inputStack[0])}")
    print(f"stride stack length 2 {len(imageStack[0][::stride])}")

    yoloModel = yolo.InitModel("ImageSegmentationAndPoseEstimation/YoloUltralyticsLib/Models/yolov8x-seg.pt")

    # Need to send yolo segmented images to dwpose model
    results = yolo.YOLOSegment(yoloModel, inputStack)

    segmentedImageStack, Bboxes = MaskSegment(inputStack, results) if useMasks else BboxSegment(inputStack, results)

    # *** THIS IS WHAT NEEDS TO BE CHANGED TO IMPLEMENT A NEW POSE MODEL ***

    # Loads the models specific config file
    config = poseModel.LoadConfig()

    # This function initialises and return a model with a weight and config path
    model = poseModel.InitModel(config)

    # This function runs the model and gets a result in the format
    # Array of inputs (multiple videos) -> frames (from one video) -> array of keypoints (for one frame)

    # Required output format:
    # [[[x, y], [x, y], ...], [[x, y], [x, y], ...], ...], [[x, y], [x, y], ...], [[x, y], [x, y], ...], ...]]
    keyPoints = poseModel.Inference(model, segmentedImageStack, config)

    # This function takes in numerous inputs and outputs the keypoint positions of selected keypoints (A potential subset of the models potential keypoints)
    # INPUTS:
    #   inputStack      : array of images         [[frames], [frames], ...]
    #   keyPointStack   : array of keypoints      [[keypointsframe1, keypointsframe2, ...], [keypointsframe1, keypointsframe2, ...], ...]
    #   bboxStack       : array of bounding boxes [[bboxframe1, bboxframe1, ...], [bboxframe1, bboxframe1, ...], ...]
    #   stride          : int value detemining the stride of images (10 would indicate the model generated keypoints for every 10th frame)
    #   drawKeypoints   : boolean value determining wether the function should draw keypoints on the image
    #   drawBboxes      : boolean value determining wether the function should draw the bounding boxes on the image
    #   drawText        : boolean value determining wether the function should draw the text for each keypoint on the image
    selectedKeyPoints = poseModel.DrawKeypoints(imageStack, keyPoints, Bboxes, stride, True, True, True, True)

    # *** THIS IS WHAT NEEDS TO BE CHANGED TO IMPLEMENT A NEW POSE MODEL ***

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