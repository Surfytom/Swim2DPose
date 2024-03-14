from ultralytics import NAS

import torch
import cv2
import numpy as np

from super_gradients.training import models
from super_gradients.common.object_names import Models

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

images, strides = LoadMediaPath("Cohoon, Start, Freestyle, 11_01_2024 11_23_43_5_Edited.mp4")

yolo_nas_pose = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").cuda()

image = cv2.cvtColor(images[25], cv2.COLOR_BGR2RGB)

results = yolo_nas_pose.predict(image, iou=0.3, conf=0.75)

results.show(show_confidence=True)