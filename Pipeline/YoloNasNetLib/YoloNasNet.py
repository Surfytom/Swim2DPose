import cv2
import numpy as np
import torch
import json
import pathlib

from super_gradients.training import models

def LoadConfig(args):
    with open(f"{pathlib.Path(__file__).parent.resolve()}/config.json", "r") as f:
        config = json.load(f)
        
    return config

def InitModel(config):
    yolo_nas_pose = models.get(config["model"], pretrained_weights=config["pretrained_weights"])

    yolo_nas_pose = yolo_nas_pose.cuda() if torch.cuda.is_available() else yolo_nas_pose

    return yolo_nas_pose

def Inference(model, imageStack, bboxes, config):
    # Runs the model on a set of images returning the keypoints the model detects
    allKeyPoints = []

    for i, images in enumerate(imageStack):
        # Loops through the
        
        allKeyPoints.append([])
        for j, image in enumerate(images['images']):
            bbox = bboxes[i][j]
            results = model.predict(image[bbox[1]:bbox[3], bbox[0]:bbox[2]], iou=config["iou"], conf=config["confidence"])

            #keyPoints = np.array(results.prediction.poses[:, :2]).astype(np.intp)
            
            allKeyPoints[i].append(results.prediction)

    return allKeyPoints

def DrawKeypoints(inputStack, keyPointStack, bboxStack, selectedKeyPoints, stride=1, drawKeypoints=True, drawBboxes=True, drawText=True, drawEdges=True):

    selectedPoints = selectedKeyPoints

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

            for p in range(loopLength):
                #print(f"p value: {(j*stride)+p}")

                #print(f"keypoint array: {keyPointArray}")

                if len(keyPointArray.poses) != 0:

                    isDrawn = np.zeros((keyPointArray.poses[0].shape[0], 1), dtype=np.uintp)

                    for t, (keyX, keyY, keyZ) in enumerate(keyPointArray.poses[0]):

                        if t in selectedPoints:
                            if drawText:
                                cv2.putText(images[(j*stride)+p], f"{t}", ((x+int(keyX))-10, (y+int(keyY))-10), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
                            if drawKeypoints:
                                cv2.circle(images[(j*stride)+p], (x + int(keyX), y + int(keyY)), 3, np.array(keyPointArray.keypoint_colors[t]).tolist(), -1)
                                isDrawn[t] = 1
                            if drawBboxes:
                                cv2.rectangle(images[(j*stride)+p], (x, y), (x1, y1), (0, 0, 255), 2)

                            selectedKeyPoints[i][j].append([(x+keyX), (y+keyY)])
                    if drawEdges:
                        for k, (origin, dest) in enumerate(np.array(keyPointArray.edge_links)):
                            #print(f"{keyPointArray.poses[0][origin][:2].astype(np.uintp)} | {keyPointArray.poses[0][dest][:2].astype(np.uintp)} | {keyPointArray.edge_colors[k]}")
                            if isDrawn[origin] and isDrawn[dest]:

                                #print(f"x, y: {x} {y} line before: {keyPointArray.poses[0][origin][:2].astype(np.uintp)} after : {[x, y] + keyPointArray.poses[0][origin][:2].astype(np.uintp)} {type([x, y] + keyPointArray.poses[0][origin][:2].astype(np.uintp))}")
                                cv2.line(images[(j*stride)+p], ([x, y] + keyPointArray.poses[0][origin][:2].astype(np.uintp)).astype(np.uintp), ([x, y] + keyPointArray.poses[0][dest][:2].astype(np.uintp)).astype(np.uintp), np.array(keyPointArray.edge_colors[k]).tolist(), 3)

    return selectedKeyPoints