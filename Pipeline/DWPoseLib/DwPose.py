import torch
import cv2
import numpy as np
import json

from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

# This script manages everything to do with the dw pose model

register_all_modules()

def LoadConfig(currentPath):
    with open(f"{currentPath}/DWPoseLib/config.json", "r") as f:
        config = json.load(f)
        config['currentPath'] = currentPath
        print('config: ', config)
    return config

def InitModel(config):
    # Simply creates and return a model based on a weight and config file path
    configPath = config["modelConfigPath"]
    weightPath = config["modelWeightPath"]
    currentPath = config['currentPath']
    return init_model(f'{currentPath}/DWPoseLib/{configPath}', f'{currentPath}/DWPoseLib/{weightPath}', device='cuda:0' if torch.cuda.is_available() else 'cpu')  # or device='cuda:0'
    
def Inference(model, imageStack, bboxes, config):
    # Runs the model on a set of images returning the keypoints the model detects
    allKeyPoints = []

    for i, images in enumerate(imageStack):
        # Loops through the
        
        allKeyPoints.append([])
        for j, image in enumerate(images['images']):
            bbox = bboxes[i][j]
            results = inference_topdown(model, image[bbox[1]:bbox[3], bbox[0]:bbox[2]])

            keyPoints = results[0].pred_instances.keypoints[0]

            keyPoints = np.array(keyPoints).astype(np.intp)
            
            allKeyPoints[i].append(keyPoints)

    return allKeyPoints

def GetBGRColours():

    colours = [
        (75, 25, 230), (75, 180, 60), (25, 225, 255), (216, 99, 67), (49, 130, 245),
        (180, 30, 145), (244, 212, 66), (230, 50, 240), (69, 239, 191), (212, 190, 250),
        (144, 153, 70), (255, 190, 220), (36, 99, 154), (200, 250, 255), (0, 0, 128),
        (195, 255, 170), (0, 128, 128), (177, 216, 255), (117, 0, 0), (169, 169, 169),
        (255, 255, 255), (0, 0, 0), (128, 0, 0),
        (255, 0, 128), (0, 255, 128)
    ]

    for colour in colours:
        yield colour

def DrawKeypoints(inputStack, keyPointStack, bboxStack, selectedKeyPoints, stride=1, drawKeypoints=True, drawBboxes=True, drawText=True, drawEdges=True):

    selectedPoints = selectedKeyPoints["keypoints"]
    edgeLinks = selectedKeyPoints["links"]

    selectedPoints.append(133)
    selectedPoints.append(134)

    selectedKeyPoints = []
    
    for i in range(len(bboxStack)):

        images = inputStack[i]['images']
        keyPoints = keyPointStack[i]
        bboxes = bboxStack[i]

        selectedKeyPoints.append([])

        for j in range(len(bboxes)):

            keyPointArray = keyPoints[j]
            box = bboxes[j]

            selectedKeyPoints[i].append([])

            x, y, x1, y1 = box

            loopLength = stride if (j*stride)+stride < len(images) else len(images) - (j*stride)

            #print(f"loop length: {loopLength}")

            chestMidPoint = (keyPointArray[5] + keyPointArray[6]) // 2
            hipMidPoint = (keyPointArray[11] + keyPointArray[12]) // 2

            keyPointArray = np.append(keyPointArray, [chestMidPoint, hipMidPoint], axis=0)

            for p in range(loopLength):
                #print(f"p value: {(j*stride)+p}")

                colourGen = GetBGRColours()    

                for t, (keyX, keyY) in enumerate(keyPointArray):
                    if t in selectedPoints:
                        if drawText:
                            cv2.putText(images[(j*stride)+p], f"{t}", ((x+keyX)-10, (y+keyY)-10), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
                        if drawKeypoints:
                            cv2.circle(images[(j*stride)+p], (x+keyX, y+keyY), 3, next(colourGen), -1)
                        if drawBboxes:
                            cv2.rectangle(images[(j*stride)+p], (x, y), (x1, y1), (0, 0, 255), 2)

                        if t != 133 and t != 134:
                            selectedKeyPoints[i][j].append([(x+keyX), (y+keyY)])

                colourGen = GetBGRColours()

                if drawEdges:
                    for edgeGroup in edgeLinks:
                        for point1, point2 in edgeLinks[edgeGroup]:
                            cv2.line(images[(j*stride)+p], ([x, y] + keyPointArray[selectedPoints[point1]]), ([x, y] + keyPointArray[selectedPoints[point2]]), next(colourGen), 3)

    return selectedKeyPoints