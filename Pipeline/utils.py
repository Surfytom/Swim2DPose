import cv2
import numpy as np
import os
import glob
import LabelBoxApi as labelBox

DEBUG = True

def GetFileNames(directory):
    file_names = []
    file_name_without_extensions = []

    files = []

    if type(directory == str):
        # List all files and directories in the specified directory
        files = glob.glob(f"{directory}/*")
    elif type(directory == list):
        files = directory
    for file in files:
        # Check if the path is a file (not a directory) or a file name
        if os.path.isfile(file) or type(file == str):
            file_name_without_extension, _ = os.path.splitext(file)
            if (_ == ".avi" or _ == ".mp4" or _ == ".png" or _ == ".jpeg"):
                file_names.append(file)
                file_name_without_extensions.append(file_name_without_extension)
    return file_names, file_name_without_extensions

# Problems when results from yolo model returns None current fix relies on first frame having a value to copy
 
def LoadMediaPath(path, stride=1):
    # Loads media (image or video) from specified path

    # Initializes images array to store images/frames
    images = []

    # Initializes array to only store every N images based on stide input
    strideImages = []

    # Only processes videos with .avi/.mp4 extension (this can be changed if needed and supported by opencv)
    if ".avi" in path or ".mp4" in path:

        # Initializes video capture strean with path to video
        videoReader = cv2.VideoCapture(path)

        # Loops through each frames appending to images each loop and appending to strideImages every stride loops
        count = 0
        while True:

            ret, frame = videoReader.read()

            if not ret:
                break

            if count % stride == 0:
                strideImages.append(frame)

            images.append(frame)

            count += 1
    else:
        # Assumes if not .mp4 or .avi that it is image

        image = cv2.imread(path)

        images.append(image)
        strideImages.append(image)

    return [images, strideImages]

def FitToImage(xyxy, imageShape):    

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
    # Takes in a bounding box top left and bottom right xy cordinate and pads it by an input amount

    if DEBUG:
        print(f"boxes before paddings: {xyxy}")

    xyxy[:2] -= padding
    xyxy[2:] += padding

    if DEBUG:
        print(f"boxes after paddings: {xyxy}")

    return xyxy

def SizeOfBox(bbox):
    # Function only used for max() sort and return size of bounding box
    bbox = bbox[1]
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def BboxSegment(imageStack, results):
    # Crops each image based on largest bounding box

    segmentedImages = []

    returnBboxes = []

    count = 0

    for images, boxes in zip(imageStack, results):
        segmentedImages.append([])
        returnBboxes.append([])
        for image, box in zip(images, boxes):
            # box contains all bounding boxes returned by segmentation model for this image

            # Finds the bounding box with the biggest area
            xyxy = box[0][max(enumerate(box[0]), key=SizeOfBox)[0]]

            # Pads box by 10 pixels
            xyxy = padBox(xyxy, 10)

            x, y, x1, y1 = FitToImage(xyxy, box[2][:2])

            # Append new added bounding box
            returnBboxes[count].append([x, y, x1, y1])

            # Append cropped image using the padded bounding box
            segmentedImages[count].append(image[y:y1, x:x1])

        count += 1
        
    return [segmentedImages, returnBboxes]

def ContourArea(contour):
    # Rturens area of a contour with a certain upper threshold (for max() sorting)

    threshold = 100000.0

    area = cv2.contourArea(contour)

    if DEBUG:
        print(f"CONTOUR AREA: {area}")

    if area > threshold:
        return 0.0
    
    return area

def MaskSegment(imageStack, results):
    # Similar to bboxSegment function but for contour masks

    segmentedImages = []
    bboxes = []

    for i in range(0, len(imageStack)):

        images = imageStack[i]

        segmentedImages.append([])
        bboxes.append([])

        for j in range(0, len(images)):

            image = images[j]

            # Initializes a black grayscale mask which is the shape of the image
            mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)

            # If results are None it means no masks were returned by the segmentation model
            if results[i][j][1] == None:
                
                # Makes mask a 100 by 100 pixel square with an origin of 0, 0
                if i == 0 and j == 0:
                    x, y = 0
                    w, h = 100

                # Applies mask to original coloured images to crop it
                shownImage = cv2.bitwise_and(image, image, mask=mask)

                segmentedImages[i].append(shownImage[y:y+h, x:x+w])
                bboxes[i].append([x, y, x+w, y+h])

                continue

            #mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)

            # converts each contour into opencv compatible contour using numpy 32 bit ints as dtype
            contours = [np.array(points, dtype=np.int32) for points in results[i][j][1]]

            # Finds the largest contour based on Contourarea function above
            largestContour = max(contours, key=ContourArea)

            # Draws the largest contour to black grayscale mask using white colour. -1 means filled
            cv2.drawContours(mask, [largestContour], 0, (255), -1)

            # Thresholds the mask into binary so black is 0 and white is 1
            mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)[1]

            # Initializes 11 x 11 kernal for open morphology function
            kernalSize = 11
            kernal = np.ones((kernalSize, kernalSize), np.uint16)

            # Performs dilation on mask making the white area expand (essentially same of bbox padding but for contours)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernal, 10)

            # Crops the orignal coloured images using mask
            shownImage = cv2.bitwise_and(image, image, mask=mask)

            # Finds the closest bounding rect to the largest contour
            x, y, w, h = cv2.boundingRect(largestContour)

            # Pads the bounding rect to account for dilation
            xyxy = padBox(np.array([x, y, x+w, y+h]), 10)

            x, y, x1, y1 = FitToImage(xyxy, mask.shape)

            if DEBUG:
                cv2.drawContours(image, [largestContour], -1, (255, 255, 255), 2)

                #cv2.drawContours(imageStack[i][j], contours, -1, (255, 255, 255), 2)

            segmentedImages[i].append(shownImage[y:y1, x:x1])
            cv2.imwrite("imageWithContour.png", image)
            #segmentedImages[i].append(shownImage)
            bboxes[i].append([x, y, x1, y1])

    return [segmentedImages, bboxes]

def GetBGRColours():
    # Defined the unique colours in bgr values for the keypoints for dw pose specifically

    colours = [
        (75, 25, 230), (75, 180, 60), (25, 225, 255), (216, 99, 67), (49, 130, 245),
        (180, 30, 145), (244, 212, 66), (230, 50, 240), (69, 239, 191), (212, 190, 250),
        (144, 153, 70), (255, 190, 220), (36, 99, 154), (200, 250, 255), (0, 0, 128),
        (195, 255, 170), (0, 128, 128), (177, 216, 255), (117, 0, 0), (169, 169, 169),
        (255, 255, 255), (0, 0, 0), (128, 0, 0)
    ]

    for colour in colours:
        yield colour

def DrawKeypoints(inputStack, keyPointStack, bboxeStack, selectedPoints, stride=1, draw=True, drawBboxes = False, drawText = False):
    # Deprecated: now use model specific draw keypoints function

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

def SaveImages(imageStack, fps, poseModel, folderPath="./results"):
    # This saves an images stack to a specific output folder

    # Gets all folders in output folder path with run in the name
    outputPaths = glob.glob(f"{folderPath}/*run*")

    max = 0

    # Loops exisitng *run* folders to find the one with the highest number
    for path in outputPaths:

        stringDigit = path[path.index("run")+3:]

        if stringDigit.isdigit():

            digit = int(stringDigit)

            if digit > max:
                max = digit

    max += 1

    # Makes a new folder wtih the new highest run digit
    os.mkdir(f"{folderPath}/{poseModel}run{max}")

    # Loops through each images or video in the stack outputting to an appropriate folder
    for i, images in enumerate(imageStack):

        os.mkdir(f"{folderPath}/{poseModel}run{max}/ouput{i}")

        if len(images) == 1:
            cv2.imwrite(f"{folderPath}/{poseModel}run{max}/ouput{i}/image.jpg", images[0])
        else:
            shape = np.shape(images[0])
            out = cv2.VideoWriter(f'{folderPath}/{poseModel}run{max}/ouput{i}/video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (shape[1], shape[0]), True)

            for image in images:
                out.write(image)

            out.release()

    print(f"Outputs saved to {folderPath}/{poseModel}run{max}")

def SaveVideoAnnotationsToLabelBox(apiKey, datasetKeyorName, datasetExisting, projectKeyorName, projectExisting, videoPaths, frameKeyPoints):

    client = labelBox.InitializeClient(apiKey)

    dataset = labelBox.InitializeDataset(client, datasetKeyorName, existing=datasetExisting)

    labelBox.AddToDataset(dataset, videoPaths)

    project = labelBox.InitializeProject(client, projectKeyorName, videoPaths, existing=projectExisting)

    labelBox.AddAnnotations(client, project.uid, videoPaths, frameKeyPoints)
