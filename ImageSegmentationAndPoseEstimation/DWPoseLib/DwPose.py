import torch
import cv2
import numpy as np
import json

from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

# This script manages everything to do with the dw pose model

register_all_modules()

def LoadConfig():
    with open("ImageSegmentationAndPoseEstimation/DWPoseLib/config.json", "r") as f:
        config = json.load(f)
        
    return config

def InitModel(config):
    # Simply creates and return a model based on a weight and config file path
    return init_model(config["modelConfigPath"], config["modelWeightPath"], device='cuda:0' if torch.cuda.is_available() else 'cpu')  # or device='cuda:0'
    
def Inference(model, imageStack, config):
    # Runs the model on a set of images returning the keypoints the model detects
    allKeyPoints = []

    for i, images in enumerate(imageStack):
        # Loops through the
        
        allKeyPoints.append([])
        for image in images:
            results = inference_topdown(model, image)

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
        (255, 255, 255), (0, 0, 0), (128, 0, 0)
    ]

    for colour in colours:
        yield colour

def DrawKeypoints(inputStack, keyPointStack, bboxStack, stride=1, drawKeypoints=True, drawBboxes=True, drawText=True):

    with open("keypointGroupings.json", "r") as f:
        keypointGroups = json.load(f)

    selectedPoints = keypointGroups["DWPose"]

    selectedKeyPoints = []
    
    for i in range(len(bboxStack)):

        images = inputStack[i]
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

                        selectedKeyPoints[i][j].append([(x+keyX), (y+keyY)])

    return selectedKeyPoints

if __name__ == "__main__":

    print(LoadConfig())

    # config_file = "DWPoseLib/256x192DWPoseModelConfig.py"

    # checkpoint_file = 'DWPoseLib/256x192DWPoseModelModel.pth'
    # model = init_model(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'

    # # please prepare an image with person
    # results = inference_topdown(model, 'input_image.jpeg')

    # keyPoints = results[0].pred_instances.keypoints[0]

    # keyPoints = np.array(keyPoints).astype(np.intp)

    # image = cv2.imread("input_image.jpeg")

    # shape = np.shape(image)

    # for x, y in keyPoints:
    #     cv2.circle(image, (x, y), 15, (0, 0, 255), -1)

    # cv2.imshow("test1", cv2.resize(image, ((shape[0] // 6), (shape[1] // 6))))

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()